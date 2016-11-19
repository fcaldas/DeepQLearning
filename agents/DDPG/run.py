# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:37:50 2016

@author: fcaldas
"""

import numpy as np
import tensorflow as tf
import gym
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit
import sys

# REPLAY BUFFER CONSTS
BUFFER_SIZE = 1000
BATCH_SIZE = 8
# FUTURE REWARD DECAY
GAMMA = 0.99
# TARGET NETWORK UPDATE STEP
TAU = 0.005
# LEARNING_RATE
LRA = 0.001
LRC = 0.01
#ENVIRONMENT_NAME
if(len(sys.argv) > 1):
    ENVIRONMENT_NAME = sys.argv[1]
else:
    ENVIRONMENT_NAME = "CartPoli-v1"

# L2 REGULARISATION
L2C = 0.00
L2A = 0.0

env = gym.make(ENVIRONMENT_NAME)
action_dim = env.action_space.shape[0]
action_high = +1.
action_low = -1.

input_dim = env.observation_space.shape[0]

sess = tf.InteractiveSession(config=tf.ConfigProto(
                             intra_op_parallelism_threads=2))
actor = ActorNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRA, L2A)
critic = CriticNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRC, L2C)
buff = ReplayBuffer(BUFFER_SIZE)
# exploration = OUNoise(action_dim)

#env.monitor.start('experiments/' + 'cartPoli-v0',force=True)

reward_vector = np.zeros(10000)

for ep in range(10000):
    # open up a game state
    s_t, r_0, done = env.reset(), 0, False
        
    #s_t = s_t.reshape()
    REWARD = 0
    # exploration.reset()
    for t in range(1000):
        if(done):
            break
        env.render()
        # select action according to current policy and exploration noise
        s_t = s_t.reshape([4])        
        #print s_t.shape        
        a_t = actor.predict([s_t]) + (np.random.randn(action_dim)/(ep / 5))
        #action_value = a_t[0]
        a_t[0,0] = np.max([-0.8, a_t[0,0]])
        a_t[0,0] = np.min([0.8, a_t[0,0]])

        # execute action and observe reward and new state
        s_t1, r_t, done, info = env.step(a_t[0])
        # store transition in replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, done)
        # sample a random minibatch of N transitions (si, ai, ri, si+1) from replay buffer
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        # set target yi = ri + gamma*target_critic_network(si+1, target_actor_network(si+1))
        #print "Debuggg"
        #print len(batch)
        #print new_states.shape
        new_states = new_states.reshape([len(batch), new_states.shape[1]])
        s_t1.reshape([s_t1.shape[0], s_t1.shape[1]])        
        #print new_states.shape        
        #print states.shape
        #print s_t1.shape        
        
        target_q_values = critic.target_predict(new_states, actor.target_predict(new_states))

        y_t = []
        for i in range(len(batch)):
            if dones[i]:
                y_t.append(rewards[i])
            else:
                y_t.append(rewards[i] + GAMMA*target_q_values[i])

        # update critic network by minimizing los L = 1/N sum(yi - critic_network(si,ai))**2
             
        y_t = np.array(y_t).reshape([len(y_t), 1])
        
        
        critic.train(y_t, states, actions)

        # update actor policy using sampled policy gradient
        a_for_grad = actor.predict(states)
        grads = critic.gradients(states, a_for_grad)
        actor.train(states, grads)

        # update the target networks
        actor.target_train()
        critic.target_train()

        # move to next state
        s_t = s_t1
        REWARD += r_t
    reward_vector[ep] = REWARD
    print "EPISODE ", ep, "ENDED UP WITH REWARD: ", REWARD

np.save("rewards.npy", reward_vector)

# Dump result info to disk
#env.monitor.close()
