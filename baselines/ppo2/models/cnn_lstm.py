import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, conv_to_fc
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample

from . import classes
from .base import Model


class CNN_LSTM(Model):
    dense_units = 256
    lstm_units = 256

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        super().__init__(sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=reuse)
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        nlstm = self.lstm_units
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            X = tf.cast(X, tf.float32)
            h = conv(X, 'c1', nf=16, rf=3, stride=1, pad='SAME', init_scale=np.sqrt(2))
            h = tf.nn.relu(h)
            h = conv(h, 'c2', nf=32, rf=3, stride=1, pad='SAME', init_scale=np.sqrt(2))
            h = tf.nn.relu(h)
            h = conv_to_fc(h)
            h = fc(h, 'fc1', nh=self.dense_units, init_scale=np.sqrt(2))
            h = tf.nn.relu(h)

            # lstm
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)

            pi_logits = fc(h5, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h5, 'q', nact)

        self.a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        self.snew = snew
        self.X = X
        self.M = M
        self.S = S
        self.pi = pi  # actual policy params now
        self.q = q
        self.sess = sess

    def step(self, ob, state, mask, *args, **kwargs):
        # returns actions, mus, states
        a0, pi0, s = self.sess.run([self.a, self.pi, self.snew], {self.X: ob, self.S: state, self.M: mask})
        return a0, pi0, s


classes.register(CNN_LSTM)
