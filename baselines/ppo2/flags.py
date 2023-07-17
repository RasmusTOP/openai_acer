import configparser
import warnings
import ast
import math

class PPOFlags:

    CFG_sections = {'RNG', 'Training', 'Log'}

    def __init__(self):

        # RNG seed
        self.seed = 0

        # Whether to use gpu
        self.use_gpu = True

        # Policy model to use
        self.policy = 'CNN'

        # Number of environments to run in parallel
        self.num_env = 16
        # Number of steps to take in each environment
        self.num_steps = 128
        # Total number of training steps
        self.total_timesteps = 1000000
        # Discount factor for rewards
        self.gamma = 0.99
        # Value function coefficient
        self.vf_coef = 0.5
        # Entropy coefficient
        self.ent_coef = 0.01
        # Learning rate for the policy network
        self.lr_policy = 0.00025
        # Learning rate for the value network
        self.lr_value = 0.00025
        # Clip range for PPO updates
        self.clip_range = 0.2
        # Number of epochs to use for each update
        self.num_epochs = 4
        # Batch size for each update
        self.batch_size = 32

        self.nminibatches = 32
        self.lam = 0.95
        self.noptepochs = 10
        self.ent_coef = 0.01
        self.lr = 7e-4
        self.cliprange = 0.1
        self.total_timesteps = 100000000
        self.max_grad_norm = 10

        # Logging directory
        self.log_dir = 'save'
        # Logging interval in number of batches
        self.log_interval = 100
        # Custom stats interval in number of batches
        self.stats_interval = 1
        # Save directory
        self.save_dir = 'save'
        # Saving interval in number of batches
        self.save_interval = 100
        # Permanently keep a checkpoint every n hours
        self.permanent_save_hours = 12

    def __str__(self):
        avoid_attr = {'from_cfg', 'CFG_sections'}
        string = self.__class__.__name__  + '('
        string += ', '.join('%s=%s' % (attr, getattr(self, attr))
                            for attr in dir(self) if not attr in avoid_attr and not attr.startswith('__'))
        string += ')'
        return string


    @classmethod
    def from_cfg(cls, path):
        config = configparser.ConfigParser()
        config.read(path)

        flags = cls()

        for sec in config.sections():
            if sec not in cls.CFG_sections:
                warnings.warn('Unrecognized section [%s]' % sec)
                continue
            for key, val in config.items(sec):
                if not hasattr(flags, key):
                    warnings.warn('Unrecognized [%s] flag in "%s" with value: %s' % (sec, key, val))
                else:
                    cast_type = type(getattr(flags, key))
                    try:
                        if cast_type == int:
                            casted_val = round(float(val))
                        elif cast_type == bool:
                            if val == 'False':
                                casted_val = False
                            elif val == 'True':
                                casted_val = True
                            else:
                                raise ValueError('bool string is not either "False" or "True"')
                        else:
                            casted_val = cast_type(val)
                    except ValueError:
                        raise ValueError('[%s] flag "%s" has an incompatible value: %s. Expected %s.'
                                         % (sec, key, val, cast_type))
                    setattr(flags, key, casted_val)

        return flags
"""
    def from_cfg(cls, path):
        config = configparser.ConfigParser()
        config.read(path)

        flags = cls()

        for sec in config.sections():
            if not sec in cls.CFG_sections:
                warnings.warn('Unrecognized section [%s]' % sec)
                continue
            for key, val in config.items(sec):
                if not hasattr(flags, key):
                    warnings.warn('Unrecognized [%s] flag in "%s" with value: %s' % (sec, key, val))
                else:
                    cast_type = type(getattr(flags, key))
                    try:
                        if cast_type == int:
                            warnings.warn('Unrecognized section 1')
                            casted_val = round(float(val))
                        elif cast_type == bool:
                            warnings.warn('Unrecognized section 2')
                            if val == 'False':
                                casted_val = False
                            elif val == 'True':
                                casted_val = True
                            else:
                                raise ValueError('bool string is not either "False" or "True"')
                        else:
                            warnings.warn('Unrecognized section 3')
                            casted_val = cast_type(val)
                    except ValueError:
                        raise ValueError('[%s] flag "%s" is of incompatiple value: %s. Expected %s.'
                                         % (sec, key, val, cast_type))
                    setattr(flags, key, casted_val)

        return flags"""