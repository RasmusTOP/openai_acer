#!/usr/bin/env python3
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer import models
from baselines.common import tf_decay
from baselines.common.cmd_util import make_atari_env, atari_arg_parser

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    env = make_atari_env(env_id, num_cpu, seed)
    policy_fn = models.get(policy)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=models.registered_list(), default='CNN')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=tf_decay.types(), default='constant')
    parser.add_argument('--logdir', help ='Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)

if __name__ == '__main__':
    main()
