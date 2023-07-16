import time
import logging
import os
import signal
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common import tf_decay, tf_util
from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import find_trainable_variables
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import EpisodeStats
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance
from baselines.acer.buffer import Buffer

# remove last step
def strip(var, nenvs, nsteps, flat = False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)

def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret

# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, num_procs, flags):
        """

        :param policy:
        :param gym.Space ob_space:
        :param gym.Space ac_space:
        :param int nenvs:
        :param int num_procs:
        :param baselines.acer.flags.AcerFlags flags:
        """
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs,
                                gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * flags.nsteps

        A = tf.placeholder(tf.int32, [nbatch]) # actions
        D = tf.placeholder(tf.float32, [nbatch]) # dones
        R = tf.placeholder(tf.float32, [nbatch]) # rewards, not returns
        MU = tf.placeholder(tf.float32, [nbatch, nact]) # mu's
        eps = 1e-6

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, flags.nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, flags.nsteps + 1, flags.nstack, reuse=True)

        params = find_trainable_variables("model")
        print("Params {}".format(len(params)))
        for var in params:
            print(var)

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(flags.alpha)
        ema_apply_op = ema.apply(params)

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            print(v.name)
            return v

        with tf.variable_scope("", custom_getter=custom_getter, reuse=True):
            polyak_model = policy(sess, ob_space, ac_space, nenvs, flags.nsteps + 1, flags.nstack, reuse=True)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i
        v = tf.reduce_sum(train_model.pi * train_model.q, axis = -1) # shape is [nenvs * (nsteps + 1)]

        # strip off last step
        f, f_pol, q = map(lambda var: strip(var, nenvs, flags.nsteps), [train_model.pi, polyak_model.pi, train_model.q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, A)
        q_i = get_by_index(q, A)

        # Compute ratios for importance truncation
        rho = f / (MU + eps)
        rho_i = get_by_index(rho, A)

        # Calculate Q_retrace targets
        qret = q_retrace(R, D, q_i, v, rho_i, nenvs, flags.nsteps, flags.gamma)

        # Calculate losses
        # Entropy
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, flags.nsteps, True)
        check_shape([qret, v, rho_i, f_i], [[nenvs * flags.nsteps]] * 4)
        check_shape([rho, f, q], [[nenvs * flags.nsteps, nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(flags.c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * flags.nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps) # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[nenvs * flags.nsteps, nact]]*2)
        gain_bc = tf.reduce_sum(
            logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (flags.c / (rho + eps))) * f),
            axis = 1) #IMP: This is sum, as expectation wrt f
        loss_bc= -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[nenvs * flags.nsteps]]*2)
        ev = q_explained_variance(tf.reshape(q_i, [nenvs, flags.nsteps]), tf.reshape(qret, [nenvs, flags.nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i)*0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)
        loss = loss_policy + flags.q_coef * loss_q - flags.ent_coef * entropy

        if flags.trust_region:
            g = tf.gradients(- (loss_policy - flags.ent_coef * entropy) * flags.nsteps * nenvs, f) #[nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps) #[nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - flags.delta) / (tf.reduce_sum(tf.square(k), axis=-1) + eps)) #[nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = avg_norm(k)
            avg_norm_g = avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [nenvs * flags.nsteps, 1]) * k
            grads_f = -g/(nenvs*flags.nsteps) # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(loss_q * flags.q_coef, params)
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]

            avg_norm_grads_f = avg_norm(grads_f) * (flags.nsteps * nenvs)
            norm_grads_q = tf.global_norm(grads_q)
            norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, params)

        if flags.max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, flags.max_grad_norm)
        grads = list(zip(grads, params))

        self.GS = GS = tf.train.get_global_step() or tf.train.create_global_step()
        self.GSwrapper = tf_util.VariableWrapper(GS)
        LR = tf_decay.schedule(decay=flags.lrschedule, init_lr=flags.lr,
                               global_step=GS, decay_steps=flags.total_timesteps)
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=flags.rprop_alpha, epsilon=flags.rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_opt_op]):
            _train = tf.group(ema_apply_op)

        # Ops/Summaries to run, and their names for logging
        run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
                     'norm_grads']
        if flags.trust_region:
            run_ops = run_ops + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g, avg_norm_k_dot_g,
                                 avg_norm_adj]
            names_ops = names_ops + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k', 'avg_norm_g',
                                     'avg_norm_k_dot_g', 'avg_norm_adj']

        def train(obs, actions, rewards, dones, mus, states, masks, steps):
            td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, GS: steps}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
                td_map[polyak_model.S] = states
                td_map[polyak_model.M] = masks
            return names_ops, sess.run(run_ops, td_map)[1:]  # strip off _train

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

class Runner(object):
    def __init__(self, env, model, nsteps, nstack):
        """

        :param baselines.common.vec_env.VecEnv env:
        :param Model model:
        :param int nsteps:
        :param int nstack:
        """
        self.env = env
        self.nstack = nstack
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.nc = nc
        self.nenv = nenv = env.num_envs
        self.nact = env.action_space.n
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1), nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs, dones=None):
        if dones is not None:
            self.obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs[:, :, :, :]

    def run(self):
        enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, state=self.states, mask=self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.update_obs(obs, dones)
            mb_rewards.append(rewards)
            enc_obs.append(obs)
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        enc_obs = np.asarray(enc_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks

class Acer():
    def __init__(self, runner, model, buffer, log_interval, stats_interval):
        """

        :param Runner runner:
        :param Model model:
        :param Buffer buffer:
        :param int log_interval:
        :param int stats_interval:
        """
        self.runner = runner
        self.model = model
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
        self.steps = None

        file_formatter = logging.Formatter('%(asctime)s %(message)s')
        stats_logger = logging.getLogger('stats_logger')
        stats_logger.setLevel(logging.INFO)
        # logger handlers
        stats_fh = logging.FileHandler(os.path.join(logger.get_dir(), 'results.log'))
        stats_fh.setFormatter(file_formatter)
        stats_logger.addHandler(stats_fh)

        self.stats_logger = stats_logger
        self.stats_interval = stats_interval

    def call(self, on_policy):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps
        if on_policy:
            enc_obs, obs, actions, rewards, mus, dones, masks = runner.run()
            self.episode_stats.feed(rewards, dones)
            if buffer is not None:
                buffer.put(enc_obs, actions, rewards, mus, dones, masks)
        else:
            # get obs, actions, rewards, mus, dones from buffer.
            obs, actions, rewards, mus, dones, masks = buffer.get()

        # reshape stuff correctly
        obs = obs.reshape(runner.batch_ob_shape)
        actions = actions.reshape([runner.nbatch])
        rewards = rewards.reshape([runner.nbatch])
        mus = mus.reshape([runner.nbatch, runner.nact])
        dones = dones.reshape([runner.nbatch])
        masks = masks.reshape([runner.batch_ob_shape[0]])

        names_ops, values_ops = model.train(obs, actions, rewards, dones, mus, model.initial_state, masks, steps)

        if on_policy and (int(steps/runner.nbatch) % self.log_interval == 0):
            logger.record_tabular("time", time.strftime('%m-%d %H:%M'))
            logger.record_tabular("total_timesteps", steps)
            logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
            logger.record_tabular("fph", '%.2fM' % ((steps/1e6)/((time.time() - self.tstart)/3600)))
            # IMP: In EpisodicLife env, during training, we get done=True at each loss of life, not just at the terminal state.
            # Thus, this is mean until end of life, not end of episode.
            # For true episode rewards, see the monitor files in the log folder.
            logger.record_tabular("mean_episode_length", self.episode_stats.mean_length())
            logger.record_tabular("mean_episode_reward", self.episode_stats.mean_reward())
            for name, val in zip(names_ops, values_ops):
                logger.record_tabular(name, float(val))
            logger.dump_tabular()

        if on_policy and (int(steps/runner.nbatch) % self.stats_interval == 0):
            if hasattr(self.runner.env, 'stats'):
                envs_stats = self.runner.env.stats()
                avg_stats = {}

                # keys_of_lists: for each key related to a list has a list in which the i-th element counts the number
                # of envs that has an i-th element in the list related to that key
                keys_of_lists = {}  # type: dict[str, list[int]]

                # init average stats
                for stats in envs_stats:
                    for key, val in stats.items():
                        if not key in avg_stats:
                            if isinstance(val, list):
                                avg_stats[key] = []
                                keys_of_lists[key] = []
                            else:
                                avg_stats[key] = 0

                # collect stats for each environment
                for stats in envs_stats:
                    for key, val in stats.items():
                        if isinstance(val, list):
                            avg_list = avg_stats[key]
                            counts = keys_of_lists[key]
                            len_diff = len(val) - len(counts)
                            if len_diff > 0:
                                counts.extend([0]*len_diff)
                                avg_list.extend([0]*len_diff)
                            for i, v in enumerate(val):
                                counts[i] += 1
                                avg_list[i] += v
                        else:
                            avg_stats[key] += val

                # average stats across envs
                for key, val in avg_stats.items():
                    if isinstance(val, list):
                        counts = keys_of_lists[key]
                        for i, v in enumerate(val):
                            val[i] = v / counts[i]
                    else:
                        avg_stats[key] = val / len(envs_stats)

                avg_stats['global_t'] = steps
                self.stats_logger.info(' '.join('%s=%s' % (key, val) for key, val in avg_stats.items()))


def learn(policy, env, flags):
    """

    :param policy:
    :param baselines.common.vec_env.VecEnv env:
    :param baselines.acer.flags.AcerFlags flags:
    """
    print("Running Acer Simple")
    print(flags)

    flags.total_timesteps = int(flags.total_timesteps)

    # disable gpu before creating any tensor
    if not flags.use_gpu:
        tf_util.disable_gpu()

    tf.reset_default_graph()
    set_global_seeds(flags.seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, num_procs=nenvs, flags=flags)

    runner = Runner(env=env, model=model, nsteps=flags.nsteps, nstack=flags.nstack)
    if flags.replay_ratio > 0:
        buffer = Buffer(env=env, nsteps=flags.nsteps, nstack=flags.nstack, size=flags.buffer_size)
    else:
        buffer = None
    nbatch = nenvs*flags.nsteps
    print("Saving at t1=%s" % nbatch)
    acer = Acer(runner, model, buffer, flags.log_interval, flags.stats_interval)

    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=flags.permanent_save_hours)
    checkpoint_dir = os.path.join(flags.save_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'model')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # load checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(model.sess, latest_checkpoint)
        start_steps = model.GSwrapper.get(model.sess)

        if hasattr(env, 'restore_state'):
            env.restore_state(checkpoint_dir, start_steps)
    else:
        start_steps = 0

    coordinator = tf.train.Coordinator()

    def signal_handler(signal, frame):
        if not coordinator.should_stop():
            coordinator.request_stop()
            print("Stopping training...")
        else:
            print("Stop already requested, please wait...")

    signal.signal(signal.SIGINT, signal_handler)
    print("Press CTRL+C to stop")

    acer.tstart = time.time()
    for acer.steps in range(start_steps, flags.total_timesteps, nbatch):

        # on policy training
        acer.call(on_policy=True)

        print("Saving at t2=%s" % nbatch)

        # off policy training
        if flags.replay_ratio > 0 and buffer.has_atleast(flags.replay_start):
            n = np.random.poisson(flags.replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this

        # saving
        do_save = (((acer.steps//nbatch) + 1) % flags.save_interval == 0) or coordinator.should_stop()
        if do_save:
            save_steps = acer.steps+nbatch

            print("Saving at t=%s" % save_steps)

            model.GSwrapper.set(model.sess, save_steps)
            saver.save(model.sess, save_path=checkpoint_path, global_step=save_steps)

            if hasattr(env, 'save_state'):
                env.save_state(checkpoint_dir, save_steps)

        if coordinator.should_stop():
            break

    env.close()
