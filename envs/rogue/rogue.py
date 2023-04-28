
import os
import pickle
import collections
import warnings
import numpy as np
import gym.spaces

from rogueinabox_lib.rogueinabox import RogueBox
from rogueinabox_lib.options import RogueOptions, RogueBoxOptions
from rogueinabox_lib.evaluator import AmuletLevelsRogueEvaluator
from .flags import RogueAcerFlags


EPISODES_TO_KEEP = 5


class RogueEnv(gym.Env):

    metadata = {'render.modes': ['ansi']}

    reward_range = (-np.inf, np.inf)

    # only an instance has a value for the following attributes because they depend on flags
    actions = None
    action_space = None
    observation_space = None

    @classmethod
    def register(cls, flags):
        """
        Registers Rogue as a gym environment

        :param RogueAcerFlags flags:
            registration flags
        """
        """
        gym.envs.register('Rogue-v1',
                          entry_point='envs.rogue:RogueEnv', trials=flags.episodes_for_evaluation,
                          reward_threshold=None, local_only=True,
                          kwargs=dict(flags=flags), nondeterministic=True,
                          tags=None, max_episode_steps=flags.max_episode_len,
                          max_episode_seconds=None, timestep_limit=None)"""

        gym.envs.register('Rogue-v1',
                          entry_point='envs.rogue:RogueEnv',
                          reward_threshold=None,
                          kwargs=dict(flags=flags), nondeterministic=True,
                          tags=None, max_episode_steps=flags.max_episode_len)

    def __init__(self, flags=RogueAcerFlags()):
        evaluator = AmuletLevelsRogueEvaluator(max_step_count=flags.max_episode_len,
                                               episodes_for_evaluation=flags.episodes_for_evaluation)

        self.rb = RogueBox(RogueBoxOptions(
                               rogue_options=RogueOptions(use_monsters=flags.use_monsters,
                                                          enable_secrets=flags.enable_secrets,
                                                          amulet_level=flags.amulet_level,
                                                          hungertime=flags.hungertime,
                                                          max_traps=flags.max_traps,
                                                          disable_dark_rooms=flags.disable_dark_rooms,
                                                          disable_mazes=flags.disable_mazes,
                                                          more_mazes=flags.more_mazes,
                                                          start_level=flags.start_level),
                               evaluator=evaluator,
                               state_generator=flags.state_generator,
                               reward_generator=flags.reward_generator,
                               transform_descent_action=flags.transform_descent_action,
                               refresh_after_commands=flags.refresh_after_commands))

        self.actions = flags.actions
        self.action_space = gym.spaces.Discrete(len(self.actions))
        state_shape = self.rb.state_generator.get_shape()
        self.observation_space = gym.spaces.Box(low=0, high=32, shape=state_shape, dtype=np.float)

        self._saved_episodes = collections.deque()

    def seed(self, seed=None):
        self.rb.rogue_options.set_seed(seed)
        return [seed]

    def render(self, mode='ansi'):
        if mode != 'ansi':
            raise NotImplementedError
        return '\n'.join(self.rb.screen) + '\n'

    def reset(self):
        self.rb.reset()
        return self.rb.state

    def close(self):
        self.rb.stop()

    def step(self, action):
        """
        :param int action:
            action to perform, represented as an integer in [0, len(actions)[

        :return:
            next_state, reward, done, info
        """
        try:
            command = self.actions[action]
        except IndexError:
            raise IndexError('Invalid rogue action index: %s. Action list has size: %s.'
                             % (action, len(self.actions)))
        reward, new_state, won, lost = self.rb.send_command(command)
        return new_state, reward, won or lost, self.rb.get_last_frame()

    def stats(self):
        return self.rb.evaluator.statistics()

    def save_state(self, path, id):
        self._save_episodes(path, id)

    def restore_state(self, path, id, warn=True):
        self._restore_episodes(path, id, warn=warn)

    def _episodes_path(self, checkpoint_dir, id):
        return os.path.join(checkpoint_dir, 'episodes', 'episodes-%s.pkl' % id)

    def _save_episodes(self, checkpoint_dir, id):
        os.makedirs(os.path.join(checkpoint_dir, 'episodes'), exist_ok=True)
        path = self._episodes_path(checkpoint_dir, id)
        with open(path, mode='wb') as pkfile:
            pickle.dump(self.rb.evaluator.episodes, pkfile)
        self._saved_episodes.append(path)
        if len(self._saved_episodes) > EPISODES_TO_KEEP:
            old_path = self._saved_episodes.popleft()
            try:
                os.unlink(old_path)
            except FileNotFoundError:
                warnings.warn('Attempting to delete unexisting episodes file %s: it was removed by an external program.'
                              % old_path, RuntimeWarning)

    def _restore_episodes(self, checkpoint_dir, id, warn=True):
        path = self._episodes_path(checkpoint_dir, id)
        try:
            with open(path, mode='rb') as pkfile:
                self.rb.evaluator.episodes = pickle.load(pkfile)
        except FileNotFoundError:
            if warn:
                warnings.warn('Episodes file %s not found: stats may be skewed.' % path, RuntimeWarning)
