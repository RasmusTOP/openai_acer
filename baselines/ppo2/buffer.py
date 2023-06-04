class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, env, nsteps, nstack, gamma, lam, size=50000):
        self.nenv = env.num_envs
        self.nsteps = nsteps
        self.nh, self.nw, self.nc = env.observation_space.shape
        self.nstack = nstack
        self.nbatch = self.nenv * self.nsteps
        self.size = size // (self.nsteps)  # Each loc contains nenv * nsteps frames, thus total buffer is nenv * size frames
        self.gamma = gamma
        self.lam = lam

        # Memory
        self.enc_obs = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None
        self.masks = None
        self.values = None
        self.advantages = None
        self.value_targets = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.num_in_buffer >= (frames // self.nsteps)

    def can_sample(self):
        return self.num_in_buffer > 0

    # Generate stacked frames
    def decode(self, enc_obs, dones):
        # enc_obs has shape [nenvs, nsteps + nstack, nh, nw, nc]
        # dones has shape [nenvs, nsteps, nh, nw, nc]
        # returns stacked obs of shape [nenv, (nsteps + 1), nh, nw, nstack*nc]
        nstack, nenv, nsteps, nh, nw, nc = self.nstack, self.nenv, self.nsteps, self.nh, self.nw, self.nc
        y = np.empty([nsteps + nstack - 1, nenv, 1, 1, 1], dtype=np.float32)
        obs = np.zeros([nstack, nsteps + nstack, nenv, nh, nw, nc], dtype=np.uint8)
        x = np.reshape(enc_obs, [nenv, nsteps + nstack, nh, nw, nc]).swapaxes(1,
                                                                              0)  # [nsteps + nstack, nenv, nh, nw, nc]
        y[nstack - 1:] = np.reshape(1.0 - dones, [nenv, nsteps, 1, 1, 1]).swapaxes(1, 0)  # keep
        y[:nstack - 1] = 1.0
        # y = np.reshape(1 - dones, [nenvs, nsteps, 1, 1, 1])
        for i in range(nstack):
            obs[-(i + 1), i:] = x
            # obs[:,i:,:,:,-(i+1),:] = x
            x = x[:-1] * y
            y = y[1:]
        return np.reshape(obs[:, nstack - 1:].transpose((2, 1, 3, 4)), (-1, obs.shape[0], obs.shape[1]))
    def get_mean(numbers):
        """
        Computes the mean of a list of numbers.
        """
        total = sum(numbers)
        mean = total / len(numbers)
        return mean

    my_numbers = [2, 4, 6, 8, 10]
    my_mean = get_mean(my_numbers)
    print("Mean:", my_mean)