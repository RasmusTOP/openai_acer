#!/usr/bin/env python3
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.cmd_util import arg_parser
from baselines.ppo2 import policies
from baselines.ppo2 import models

from envs import RogueEnv, RogueSubprocVecEnv, RoguePPOFlags

from baselines import bench, logger
from baselines.ppo2 import ppo2
def main():
    parser = arg_parser()
    parser.add_argument('--flags', '-f',
                        help="flags cfg file (will load checkpoint in save dir if found)",
                        default=None)
    args = parser.parse_args()

    flags = RoguePPOFlags.from_cfg(args.flags) if args.flags else RoguePPOFlags()
    RogueEnv.register(flags)
    logger.configure(flags.log_dir)

    env = make_rogue_env(num_env=flags.num_env, seed=flags.seed)

    set_global_seeds(flags.seed)
    policy_fn = models.get(flags.policy)
    ppo2.learn(policy=policy_fn, env=env, nsteps=128, nminibatches=4,
               lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
               ent_coef=.01,
               lr=lambda f: f * 2.5e-4,
               cliprange=lambda f: f * 0.1,
               total_timesteps=int(1e8 * 1.1))
'''
    ppo2.learn(policy_fn=policy_fn, env=env, total_timesteps=flags.total_timesteps, nsteps=flags.nsteps,
               nminibatches=flags.nminibatches, lam=flags.lam, gamma=flags.gamma, noptepochs=flags.noptepochs,
               log_interval=flags.log_interval, ent_coef=flags.ent_coef, lr=flags.lr, cliprange=flags.cliprange,
               save_interval=flags.save_interval, load_path=flags.load_path, tensorboard_log=flags.tensorboard_log)

    env.close()
'''

def make_rogue_env(num_env, seed=None, start_index=0):
    def make_env(rank):
        def _thunk():
            env = gym.make('Rogue-v1')
            env.seed(seed + rank)
            return env
        return _thunk
    return RogueSubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

if __name__ == '__main__':
    main()
