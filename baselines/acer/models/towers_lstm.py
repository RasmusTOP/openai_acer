import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample

from . import classes
from .base import Model


class Towers_LSTM(Model):
    dense_units = 256
    lstm_units = 256
    depth = 2

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        super().__init__(sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=reuse)
        nbatch = nenv * nsteps
        print(nenv, nsteps, "Lstm ", nbatch)
        print(nbatch)
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


classes.register(Towers_LSTM)
