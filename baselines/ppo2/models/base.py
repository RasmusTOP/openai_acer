
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, nlstm=256):
        pass

    @abstractmethod
    def step(ob, state, mask, *args, **kwargs):
        pass
