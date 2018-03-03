# build A3C network

import numpy as np
import tensorflow as tf

GAMMA = 0.90
ENTROPY_WEIGHT = 0.01
ENTROPY_EPS = 1e-6
REG_PARA = 1e-4


class ActorNetwork(object):
    """
    input is status, output is action distribution
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create actor network 
        self.inputs, self.out = self.create_actor_network()

        # Set all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # selected action, one-hot
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # this is advantage function to present gradient
        self.advantage = tf.placeholder(tf.float32, [None, 1])

        # compute loss 
        self.obj = tf.reduce_mean(tf.multiply(
            tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                 reduction_indices=1, keep_dims=True)),
            -self.advantage)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
                                                                tf.log(self.out + ENTROPY_EPS)))

        # Optimizer op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).minimize(self.obj)

    def my_dense(self, inp, units):
        reg = tf.contrib.layers.l2_regularizer(REG_PARA)
        output = tf.layers.dense(
            inputs=inp, units=units, kernel_regularizer=reg)
        output = tf.nn.relu(output)
        return output

    def padding(self, state):
        tmp = state
        tmp = tf.reshape(tmp, shape=[-1, 4, 4, 12])
        paddings = [[0, 0], [5, 5], [5, 5], [0, 0]]
        tmp = tf.pad(tmp, paddings, "CONSTANT")
        return tmp

    def create_actor_network(self):
        with tf.variable_scope("actor"):
            inputs = tf.placeholder("float", [None, self.s_dim])
            output = tf.reshape(inputs, shape=[-1, 4, 4, 12])
            output = self.padding(output)
            output = tf.layers.conv2d(output, 64, 6, activation=tf.nn.relu)
            output = tf.layers.conv2d(output, 32, 4, activation=tf.nn.relu)
            output = tf.layers.conv2d(output, 32, 2, activation=tf.nn.relu)
            output = tf.contrib.layers.flatten(output)
            output = self.my_dense(output, 32)
            output = self.my_dense(output, 32)
            output = self.my_dense(output, 16)
            output = tf.layers.dense(inputs=output, units=self.a_dim)
            output = tf.nn.softmax(output)
            print "created actor network"
            return inputs, output

    def train(self, inputs, acts, advantage):
        return self.sess.run([self.obj, self.optimize], feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.advantage: advantage
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        return self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    input network state (and may action), output V(s)
    """

    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        self.inputs, self.out = self.create_critic_network()

        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))

        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # network V(s)
        self.sampled_value_s = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.losses.mean_squared_error(self.sampled_value_s, self.out)

        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).minimize(self.loss)

    def my_dense(self, inp, units):
        reg = tf.contrib.layers.l2_regularizer(REG_PARA)
        output = tf.layers.dense(
            inputs=inp, units=units, kernel_regularizer=reg)
        output = tf.nn.relu(output)
        return output

    def padding(self, state):
        tmp = state
        tmp = tf.reshape(tmp, shape=[-1, 4, 4, 12])
        paddings = [[0, 0], [5, 5], [5, 5], [0, 0]]
        tmp = tf.pad(tmp, paddings, "CONSTANT")
        return tmp

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tf.placeholder('float', [None, self.s_dim])
            output = tf.reshape(inputs, shape=[-1, 4, 4, 12])
            output = self.padding(output)
            output = tf.layers.conv2d(output, 64, 6, activation=tf.nn.relu)
            output = tf.layers.conv2d(output, 32, 4, activation=tf.nn.relu)
            output = tf.layers.conv2d(output, 32, 2, activation=tf.nn.relu)
            output = tf.contrib.layers.flatten(output)
            output = self.my_dense(output, 32)
            output = self.my_dense(output, 32)
            output = self.my_dense(output, 16)
            output = tf.layers.dense(inputs=output, units=1)
            print "created critic network"
            return inputs, output

    def train(self, inputs, value_s):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.sampled_value_s: value_s
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)})


def learn(s_batch, a_batch, r_batch, terminal, actor, critic):
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    batch_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = r_batch[-1, 0]
    else:
        R_batch[-1, 0] = v_batch[-1, 0]

    for t in reversed(xrange(batch_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    # advantage function: Q(s,a) - v(s)
    advantage_batch = R_batch - v_batch

    actor_loss_batch, _ = actor.train(s_batch, a_batch, advantage_batch)
    critic_loss_batch, _ = critic.train(s_batch, R_batch)

    return advantage_batch, actor_loss_batch, critic_loss_batch


def build_summaries():
    advantage = tf.Variable(0.)
    tf.summary.scalar("Advantage", advantage)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar("Critic_loss", critic_loss)
    summary_vars = [advantage, eps_total_reward, critic_loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
