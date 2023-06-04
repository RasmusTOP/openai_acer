import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample
from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm

from . import classes
from .base import Model


class Towers_LSTM:
    dense_units = 256
    lstm_units = 256
    depth = 2

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
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
            with tf.variable_scope("Towers", reuse=reuse):
                with tf.variable_scope("tower_1"):
                    tower1 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                              padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
                    tower1 = tf.layers.conv2d(inputs=tower1, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                              padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
                    tower1 = tf.layers.max_pooling2d(tower1, pool_size=(22, 80), strides=(22, 80))

                with tf.variable_scope("tower_2"):
                    tower2 = tf.layers.max_pooling2d(X, pool_size=(2, 2), strides=(2, 2))
                    for _ in range(self.depth):
                        tower2 = tf.layers.conv2d(inputs=tower2, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                                  padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
                        tower2 = tf.nn.relu(tower2)
                    tower2 = tf.layers.max_pooling2d(tower2, pool_size=(11, 40), strides=(11, 40))

                with tf.variable_scope("tower_3"):
                    tower3 = tf.layers.max_pooling2d(X, pool_size=(3, 6), strides=(3, 6), padding='SAME')
                    for _ in range(self.depth):
                        tower3 = tf.layers.conv2d(inputs=tower3, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                                  padding='SAME', kernel_initializer=tf.initializers.variance_scaling)
                        tower3 = tf.nn.relu(tower3)
                    tower3 = tf.layers.max_pooling2d(tower3, pool_size=(8, 14), strides=(8, 14), padding='SAME')

                concat = tf.concat([tower1, tower2, tower3], axis=-1)

            # lstm
            xs = batch_to_seq(concat, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

classes.register(Towers_LSTM)
