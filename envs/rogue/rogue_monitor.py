
import logging
from gym.core import Wrapper


class RogueMonitor(Wrapper):

    def __init__(self, env, filename, stats_steps=850):
        super().__init__(env)

        file_formatter = logging.Formatter('%(asctime)s %(message)s')
        stats_logger = logging.getLogger(filename + '.stats_logger')
        stats_logger.setLevel(logging.INFO)
        # logger handlers
        stats_fh = logging.FileHandler(filename + '.stats.log')
        stats_fh.setFormatter(file_formatter)
        stats_logger.addHandler(stats_fh)

        self.stats_logger = stats_logger
        self.stats_step = stats_steps
        self.last_stats_log = 0

        self.total_steps = 0

    def step(self, action):
        res = self.env.step(action)
        self.total_steps += 1
        if (self.total_steps - self.last_stats_log) >= self.stats_step:
            self.last_stats_log = self.total_steps
            stats = self.env.unwrapped.stats()
            print("suck me dry")
            print(stats)
            self.stats_logger.info(' '.join('%s=%s' % (key, val) for key, val in stats.items()))
        return res

    def reset(self):
        return self.env.reset()

    def render(self, mode='ansi'):
        super().render(mode=mode)
