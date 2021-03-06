import logging
import os

import numpy as np
import tensorflow as tf

import a3c
from env2048 import GameGrid

NUM_AGENTS = 1
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
MODEL_DIR = './models'
NN_MODEL = './models/nn_model_ep_5.ckpt'
TRAIN_SEQ_LEN = 500
MODEL_SAVE_INTERVAL = 5
DISPLAY_REWARD_THRESHOLD = 100
MAX_EPISODE = 5
RENDER = True

env = GameGrid()
A_DIM = env.n_actions
MAX_NUM = env.max_num
S_INFO = int(env.n_features * (np.log2(MAX_NUM) + 1))


class Agent(object):
    def __init__(self, sess):
        self.sess = sess
        self.actor = a3c.ActorNetwork(self.sess,
                                      state_dim=S_INFO, action_dim=A_DIM,
                                      learning_rate=ACTOR_LR_RATE)
        self.critic = a3c.CriticNetwork(self.sess,
                                        state_dim=S_INFO,
                                        learning_rate=CRITIC_LR_RATE)

        self.summary_ops, self.summary_vars = a3c.build_summaries()

        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        self.saver = tf.train.Saver()

        # restore neural network
        if NN_MODEL is not None:
            print("load model success!")
            self.saver.restore(self.sess, NN_MODEL)

        self.epoch = 0
        self.i_episode = 0
        self.total_reward = 0.0

        self.s = env.reset()

    def run(self):

        while self.i_episode < MAX_EPISODE:

            s_batch, a_batch, r_batch, terminal, self.s = self.one_epoch()
            self.total_reward += np.sum(r_batch)
            if terminal:
                print("Episode {} get total reward={}".format(self.i_episode, self.total_reward))
                self.i_episode += 1
                self.s = env.reset()
                self.total_reward = 0.0

            advantage_batch, actor_loss_batch, critic_loss_batch = \
                a3c.learn(s_batch=np.stack(s_batch, axis=0),
                          a_batch=np.vstack(a_batch),
                          r_batch=np.vstack(r_batch),
                          terminal=terminal,
                          actor=self.actor, critic=self.critic)

            self.summary(advantage_batch, critic_loss_batch, r_batch)

    def one_hot(self, state):
        state = np.log2(state)
        whereinf = np.isinf(state)
        state[whereinf] = 0
        state = tf.one_hot(state, np.log2(MAX_NUM) + 1)
        state = tf.reshape(state, [-1])
        state = self.sess.run(state)
        return state

    def one_epoch(self):
        s_batch = []
        a_batch = []
        r_batch = []
        ep_rs_sum = 0

        state = self.s

        while True:
            action_vec = np.zeros(A_DIM)
            # if RENDER:
            #    #env.deiconify()
            #    env.render()

            state = self.one_hot(state)

            action_prob = self.actor.predict(np.reshape(state, (1, S_INFO)))
            # print action_prob
            action_cumsum = np.cumsum(action_prob)
            current_a = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            s_, reward, flag_end = env.step(current_a)
            r_batch.append(reward)
            s_batch.append(state)
            action_vec[current_a] = 1
            a_batch.append(action_vec)
            state = s_
            ep_rs_sum += reward

            if flag_end:
                for i in range(len(s_)):
                    if i % 4 == 0:
                        print("")
                    print("%5d" % s_[i]),
                print("")

            if len(r_batch) >= TRAIN_SEQ_LEN or flag_end:
                break

        return s_batch, a_batch, r_batch, flag_end, state

    def summary(self, advantage_batch, critic_loss_batch, r_batch):

        total_advantage = 0.0
        total_critic_loss = 0.0
        total_batch_len = 0

        total_reward = np.sum(r_batch)
        total_advantage += np.sum(advantage_batch)
        total_critic_loss += np.sum(critic_loss_batch)
        total_batch_len += len(r_batch)
        self.epoch += 1
        avg_advantage_loss = total_advantage / total_batch_len
        avg_critic_loss = total_critic_loss / total_batch_len
        summary_str = self.sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: avg_advantage_loss,
            self.summary_vars[1]: total_reward,
            self.summary_vars[2]: avg_critic_loss,
        })
        self.writer.add_summary(summary_str, self.epoch)
        self.writer.flush()
        if self.i_episode % MODEL_SAVE_INTERVAL == 0:
            save_path = self.saver.save(self.sess, MODEL_DIR + "/nn_model_ep_" +
                                        str(int(self.i_episode)) + ".ckpt")
            logging.info("Model saved in file: " + save_path)


def main():
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with tf.Session() as sess:
        agent = Agent(sess)
        agent.run()


if __name__ == "__main__":
    main()
