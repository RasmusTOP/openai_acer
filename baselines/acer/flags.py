
import configparser
import warnings


class AcerFlags:

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
        # Number of n-steps returns
        self.nsteps = 20
        # Number of frames to stack (for atari games)
        self.nstack = 4
        # Total number of training steps
        self.total_timesteps = 80e6
        # Replay buffer size
        self.buffer_size = 50000
        # Ratio of memory replay
        self.replay_ratio = 4
        # Training steps at which memory replay will begin
        self.replay_start = 10000

        # Q loss coefficinet
        self.q_coef = 0.5
        # Entropy loss coefficient
        self.ent_coef = 0.01
        # Gradients clipping ratio
        self.max_grad_norm = 10
        # Inital learning rate
        self.lr = 7e-4
        # Learning rate decay type
        self.lrschedule = 'linear'
        # RMSprop epsilon
        self.rprop_epsilon = 1e-5
        # RMSprop alpha
        self.rprop_alpha = 0.99
        # Discount value
        self.gamma = 0.99
        # Importance sampling truncated ratio
        self.c = 10.0
        # Whether to use TRPO
        self.trust_region = True
        # Average policy soft update coefficient
        self.alpha = 0.99
        # Delta as defined in ACER for TRPO
        self.delta = 1

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

        return flags
