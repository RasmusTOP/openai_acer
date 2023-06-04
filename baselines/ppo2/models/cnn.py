import tensorflow as tf
from baselines.ppo2.policies import nature_cnn
from baselines.a2c.utils import fc, sample

from .base import Model
from . import classes


class CNN(Model):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        super().__init__(sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=reuse)
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi_logits = fc(h, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h, 'q', nact)

        self.a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        self.X = X
        self.pi = pi  # actual policy params now
        self.q = q
        self.sess = sess

    def step(self, ob, *args, **kwargs):
        # returns actions, mus, states
        a0, pi0 = self.sess.run([self.a, self.pi], {self.X: ob})
        return a0, pi0, []  # dummy state

    def out(self, ob, *args, **kwargs):
        pi0, q0 = self.sess.run([self.pi, self.q], {self.X: ob})
        return pi0, q0

    def act(self, ob, *args, **kwargs):
        return self.sess.run(self.a, {self.X: ob})


classes.register(CNN)
