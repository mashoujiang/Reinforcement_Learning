import os
import sys
import time
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import random
import a3c
from maze_env import Maze
#from progbar import ProgBar

NUM_AGENTS = 1
ACTOR_LR_RATE = 0.001
CRITIC_LR_RATE = 0.01
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
MODEL_DIR = './models'
NN_MODEL = None
TRAIN_SEQ_LEN = 10
MODEL_SAVE_INTERVAL = 100
DISPLAY_REWARD_THRESHOLD = 100
MAX_EPISODE = 8000

env = Maze()
S_INFO = env.n_features
A_DIM = env.n_actions


def central_agent(net_params_queues, exp_queues):
    
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(sess,
                                state_dim=S_INFO, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                state_dim=S_INFO,
                                learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        saver = tf.train.Saver()

        # restore neural network
        nn_model = NN_MODEL
        if nn_model is not None:
            saver.restore(sess, nn_model)

        epoch = 0
	i_episode = np.zeros(NUM_AGENTS)

        while True:
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            total_reward = 0.0
            total_agents = 0.0
            total_advantage = 0.0
            total_critic_loss = 0.0
            total_batch_len = 0
            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, i_episode[i] = exp_queues[i].get()

	    	# only agent_id=0 i_episode > MAX_EPISODE, break loop.
		if terminal:
	    	    if i_episode > MAX_EPISODE:
		        print "central agent exit."
		        sys.exit(0)

                advantage_batch, actor_loss_batch, critic_loss_batch = \
                                a3c.learn(s_batch=np.stack(s_batch,axis=0),
                                          a_batch=np.vstack(a_batch),
                                          r_batch=np.vstack(r_batch),
                                          terminal=terminal,
                                          actor=actor, critic=critic)

                total_reward += np.sum(r_batch)
                total_agents += 1.0
                total_advantage += np.sum(advantage_batch)
                total_critic_loss +=np.sum(critic_loss_batch)
                total_batch_len +=len(r_batch)

            epoch +=1
            avg_reward = total_reward / total_agents
            avg_advantage_loss = total_advantage / total_batch_len
            avg_critic_loss = total_critic_loss / total_batch_len

            summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: avg_advantage_loss,
                        summary_vars[1]: avg_reward,
                        summary_vars[2]: avg_critic_loss,
                        })
            
            writer.add_summary(summary_str, epoch)
            writer.flush()

            if i_episode[0] % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(sess, MODEL_DIR + "/nn_model_ep_" + 
                                        str(int(i_episode[0])) + ".ckpt")
                logging.info("Model saved in file: " + save_path)


def agent(agent_id, net_params_queue, exp_queue):
    with tf.Session() as sess:
        actor = a3c.ActorNetwork(sess,
                                state_dim=S_INFO, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                state_dim=S_INFO,
                                learning_rate=CRITIC_LR_RATE)

        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        action_vec = np.zeros(A_DIM)
        s_batch = []
        a_batch = []
        r_batch = []

        s = env.reset()
        running_reward = 0
        RENDER = True
        ep_rs_sum = 0
        first_flag = True
        i_episode = 0

        while True:
            if agent_id ==0 and RENDER: 
                env.deiconify()
                env.render()
            
            state = s
            action_prob = actor.predict(np.reshape(state, (1, S_INFO)))
	    #print action_prob
            action_cumsum = np.cumsum(action_prob)
            current_a = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            s_, reward, flag_end = env.step(current_a)
            r_batch.append(reward)
            s_batch.append(state)
            action_vec[current_a] = 1
            a_batch.append(action_vec)
            action_vec = np.zeros(A_DIM)
            s = s_
            ep_rs_sum +=reward
            
            if len(r_batch) >= TRAIN_SEQ_LEN or flag_end:
                exp_queue.put([s_batch,
                                a_batch,
                                r_batch,
                                flag_end,
				i_episode])
		if flag_end:
		    if i_episode > MAX_EPISODE:
		        print "single agent exit"
		        sys.exit(0)	

                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                
		del s_batch[:]
                del a_batch[:]
                del r_batch[:]
	    
            if flag_end:
                if first_flag:
                    print "first init running reward"
                    first_flag = False
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True         
                #if agent_id == 0:
                #    print("episode:", i_episode, " reward:", int(running_reward))
                i_episode += 1
                ep_rs_sum = 0
                s = env.reset()
	


def main():
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    net_params_queues = []
    exp_queues = []

    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, net_params_queues[i], exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    print "agents started"
    coordinator.join()
    print "coordinator joined"

if __name__ == "__main__":
    main()
