import time
import logging
import os
import signal
import numpy as np
import tensorflow as tf

from baselines.common.runners import AbstractEnvRunner
from baselines.common.models import mlp
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common import tf_decay, tf_util
from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import find_trainable_variables
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import EpisodeStats
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance
from baselines.acer.buffer import Buffer
from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import fc

# remove last step
def strip(var, nenvs, nsteps, flat = False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)

def calculate_gae(rewards, values, dones, gamma, lambda_=0.95):
    """
    Calculate Generalized Advantage Estimation (GAE) for PPO.

    :param rewards: list of rewards
    :param values: list of value predictions
    :param dones: list of done flags
    :param gamma: discount factor
    :param lambda_: GAE parameter
    :return: list of advantages
    """
    T = len(rewards)
    deltas = np.zeros(T)
    advantages = np.zeros(T)
    last_advantage = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t+1] - values[t]
        deltas[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        last_advantage = deltas[t]

    advantages = deltas.cumsum()  # Calculate cumulative sum of advantages
    returns = advantages + values[:-1]  # Calculate the target value for the value function update

    return advantages, returns

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nbatch, nsteps, ent_coef,num_procs, vf_coef, max_grad_norm):
        """
        Initialize the model with the necessary parameters and constructs the policy and value function networks.
        """
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=num_procs,
                               inter_op_parallelism_threads=num_procs,
                               gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = sess = tf.Session(config=config)
        self.sess = tf.Session()

        # Create placeholders for observations, actions and advantages
        self.obs = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
        self.acs = tf.placeholder(tf.float32, [None] + list(ac_space.shape))
        self.advs = tf.placeholder(tf.float32, [None])

        # Create the policy and value function networks
        self.pdtype = make_pdtype(ac_space)
        self.pd, self.pi = policy(self.obs, self.pdtype.param_shape_flat())
        self.vf = fc(self.pi, 'v', 1)

        # Define the loss function
        self.loss = -tf.reduce_mean(self.pd.logp(self.acs) * self.advs) + vf_coef * tf.reduce_mean(tf.square(self.vf - self.advs))

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs, state=None, mask=None):
        """
        Takes a step in the environment using the current policy.
        """
        action, value = self.sess.run([self.pi, self.vf], feed_dict={self.obs: obs})
        return action, value, state

    def value(self, obs, state=None, mask=None):
        """
        Returns the value function's estimate of the state value for a given state.
        """
        return self.sess.run(self.vf, feed_dict={self.obs: obs})

    def save(self, save_path):
        """
        Saves the model's parameters to a file.
        """
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):
        """
        Loads the model's parameters from a file.
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)

class PPOModel(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, num_procs, flags):

        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs,
                                gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = sess = tf.Session(config=config)

        nact = ac_space.n
        nbatch = nenvs * flags.nsteps

        # Define placeholders
        self.A = tf.placeholder(tf.int32, [nbatch])  # actions
        self.D = tf.placeholder(tf.float32, [nbatch])  # dones
        self.R = tf.placeholder(tf.float32, [nbatch])  # rewards, not returns
        self.MU = tf.placeholder(tf.float32, [nbatch, nact])  # mu's
        self.LR = tf.placeholder(tf.float32, [])  # learning rate
        eps = 1e-6

        # Create policy and value networks
        self.step_model = policy(sess, ob_space, ac_space, nenvs, 1, flags.nstack, reuse=False)
        self.train_model = policy(sess, ob_space, ac_space, nenvs, flags.nsteps + 1, flags.nstack, reuse=True)

        def build_computation_graph(self):
            # Define placeholders for state and mask
            self.X = self.train_model.X
            self.S = tf.placeholder(shape=(nbatch,) + self.train_model.S.shape.as_list()[1:],
                                    dtype=self.train_model.S.dtype)
            self.M = tf.placeholder(shape=(nbatch,) + self.train_model.M.shape.as_list()[1:],
                                    dtype=self.train_model.M.dtype)

            # Get policy and value outputs from the train model
            pi, v = self.train_model.pi, self.train_model.q

            # Strip off the last step
            pi, v = strip(pi, nenvs, flags.nsteps), strip(v, nenvs, flags.nsteps)

            # Get pi and q values for actions taken
            pi_i = get_by_index(pi, self.A)
            q_i = get_by_index(v, self.A)

            # Calculate ratios for importance sampling
            ratio = pi_i / (self.MU + eps)

            # Calculate advantages using Generalized Advantage Estimation (GAE)
            advantage, returns = self.calculate_gae(self.R, v, self.D, flags.gamma, )

            # Calculate surrogate loss for policy
            surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, tf.clip_by_value(ratio, 1 - flags.epsilon,
                                                                                            1 + flags.epsilon) * advantage))

            # Calculate value function loss
            value_loss = tf.reduce_mean(tf.square(q_i - tf.reshape(v, [nenvs * flags.nsteps, 1]))) * 0.5

            # Calculate entropy regularization loss
            entropy_loss = tf.reduce_mean(cat_entropy_softmax(pi))

            # Calculate total loss
            loss = surrogate_loss - flags.ent_coef * entropy_loss + flags.vf_coef * value_loss

            # Perform gradient updates
            optimizer = tf.train.AdamOptimizer(learning_rate=self.LR)
            self.train_op = optimizer.minimize(loss)

        def train(self, obs, actions, rewards, dones, mus, states, masks, learning_rate):
            feed_dict = {
                self.X: obs,
                self.A: actions,
                self.R: rewards,
                self.D: dones,
                self.MU: mus,
                self.S: states,
                self.M: masks,
                self.LR: learning_rate
            }
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            return loss
    def act(self, obs):
        # Implement the method to perform an action based on the given observation
        pass

    def update(self, rollouts):
        # Implement the method to update the model based on collected rollouts
        pass




class PPOBuffer(object):
    def __init__(self, size):
        # Initialize the buffer with the given size
        pass

    def store(self, rollout):
        # Store a rollout in the buffer
        pass

    def get(self):
        # Get a batch of experiences from the buffer for training
        pass


class PPOPolicy(object):
    def __init__(self, model):
        # Initialize the policy with the given model
        self.model = model

    def select_action(self, obs):
        # Implement the method to select an action based on the given observation using the policy network
        pass

    def compute_log_probs(self, obs, actions):
        # Compute the log probabilities of the given actions based on the policy network
        pass

class PPOAlgorithm(object):
    def __init__(self, env, model, buffer, policy):
        # Initialize the algorithm with the given environment, model, buffer, and policy
        self.env = env
        self.model = model
        self.buffer = buffer
        self.policy = policy

    def learn(self, total_timesteps):
        # Implement the training loop of the PPO algorithm
        for t in range(total_timesteps):
            # Collect experiences using the current policy
            rollout = self.collect_rollout()

            # Store the collected rollout in the buffer
            self.buffer.store(rollout)

            # Perform policy and value function updates based on the collected rollouts
            self.model.update(self.buffer.get())


'''
# Actor-Critic Network for PPO
class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.policy_fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.policy_fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)

        self.value_fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.value_fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.value_output = tf.keras.layers.Dense(1)

    def call(self, state):
        policy = self.policy_fc1(state)
        policy = self.policy_fc2(policy)
        logits = self.policy_logits(policy)

        value = self.value_fc1(state)
        value = self.value_fc2(value)
        value = self.value_output(value)

        return logits, value


class Model:
    def __init__(self, ob_space, ac_space, nbatch, nsteps, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        sess = tf.get_default_session()

        # Create the input placeholders
        self.obs_ph = tf.placeholder(dtype=ob_space.dtype, shape=(None, *ob_space.shape), name="obs_ph")
        self.acs_ph = tf.placeholder(dtype=tf.int32, shape=(None,), name="acs_ph")
        self.returns_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="returns_ph")
        self.advs_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="advs_ph")
        self.old_probs_ph = tf.placeholder(dtype=tf.float32, shape=(None, ac_space.n), name="old_probs_ph")
        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")

        # Create the policy and value function networks
        self.policy, self.pi_logits, self.value = self._build_net(self.obs_ph, ac_space.n)

        # Define the loss functions
        self._build_losses(ac_space)

        # Define the optimization operations
        self._build_optimizers(max_grad_norm, nbatch, nsteps)

    def _build_net(self, obs, nact):
        # Define the neural network architecture for the policy and value function
        # ...
        return policy, pi_logits, value

    def _build_losses(self, ac_space):

    # Define the loss functions for the policy and value function
    # ...

    def _build_optimizers(self, max_grad_norm, nbatch, nsteps):


# Define the optimization operations for the policy and value function
# ...
def run(self, gamma=0.99, lam=0.95):
    enc_obs = np.split(self.obs, self.nstack, axis=3)
    mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards, mb_values = [], [], [], [], [], []
    for _ in range(self.nsteps):
        actions, mus, states, values = self.model.step(self.obs, state=self.states, mask=self.dones)
        mb_obs.append(np.copy(self.obs))
        mb_actions.append(actions)
        mb_mus.append(mus)
        mb_dones.append(self.dones)
        mb_values.append(values)
        obs, rewards, dones, _ = self.env.step(actions)
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
    mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)

    mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

    mb_masks = mb_dones
    mb_dones = mb_dones[:, 1:]

    last_values = self.model.value(self.obs, state=self.states, mask=self.dones)
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(self.nsteps)):
        if t == self.nsteps - 1:
            nextnonterminal = 1.0 - self.dones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_dones[:, t + 1]
            nextvalues = mb_values[:, t + 1]
        delta = mb_rewards[:, t] + gamma * nextvalues * nextnonterminal - mb_values[:, t]
        mb_advs[:, t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    mb_returns = mb_advs + mb_values

    return enc_obs, mb_obs, mb_actions, mb_returns, mb_advs, mb_mus, mb_masks


# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size, clip_ratio=0.2, gamma=0.99, lr=0.0003, beta=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.beta = beta

        self.model = ActorCritic(state_size, action_size)

    def get_action(self, state):
        logits, _ = self.model(state)
        probs = tf.nn.softmax(logits)
        action = np.random.choice(self.action_size, p=probs.numpy()[0])
        return action

    def compute_loss(self, state, action, old_probs, old_values, returns, advantage):
        logits, values = self.model(state)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)

        # Policy loss
        entropy = -tf.reduce_sum(probs * log_probs, axis=1, keepdims=True)
        new_probs = tf.gather_nd(probs, tf.stack([tf.range(len(action)), action], axis=1))
        old_probs = tf.cast(old_probs, tf.float32)
        ratio = tf.exp(tf.math.log(new_probs) - tf.math.log(old_probs))
        clipped_ratio = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))
        policy_loss -= self.beta * tf.reduce_mean(entropy)

        # Value loss
        value_loss = tf.reduce_mean(tf.square(returns - values))

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        return loss

    def train(self, states, actions, old_probs, old_values, returns, advantages):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        old_values = tf.convert_to_tensor(old_values, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss = self.compute_loss(states, actions, old_probs, old_values, returns, advantages)

        grads = tape.gradient(loss
'''