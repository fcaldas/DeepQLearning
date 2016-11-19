# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:34:21 2016

@author: fcaldas
"""

from neuralnet import *
from replaybuffer import *
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn
import sys

REPLAY_BUFFER_SIZE = 16


if(len(sys.argv) >= 2 and sys.argv[1] == "CartPole-v1"):
    env = gym.make("CartPole-v1")
    envname = "CartPole1"
    actor = NNet(REPLAY_BUFFER_SIZE, actions=np.array([0,1]))
elif(len(sys.argv) >= 2 and sys.argv[1] == "CartPoli-v1"):
    env = gym.make('CartPoli-v1')
    envname = "CartPole1"
    actor = NNet(REPLAY_BUFFER_SIZE, actions=np.linspace(-1,1,num=7))
elif(len(sys.argv) >= 2 and sys.argv[1] == 'CartPoli-v2'):
    env = gym.make("CartPoli-v2")
    envname= "CartPole2"
    actor = NNet(REPLAY_BUFFER_SIZE, actions=np.linspace(-.7,.7,num=30))
else:
    envname = "CartPole0"
    env = gym.make("CartPoli-v0")
    actor = NNet(REPLAY_BUFFER_SIZE, actions=np.linspace(-1,1,num=50))
   
if(len(sys.argv) == 3 and sys.argv[2] == "norender"):
    norender = True
elif(len(sys.argv) == 4 and sys.argv[2] == "load"):
    actor.load(sys.argv[3])
    norender = False
else:
    norender = False

env.monitor.start('./' + envname, force=True)

episodes = 5000
reward_eps = np.zeros(episodes)
replay = ReplayBuffer(1400)
SCORE_THOLD = 4400
n_episodes_thold = 0
do_train = True

for ep in xrange(0, episodes):
    s_t, r_t, done = env.reset(), 0, False
    #s_t = s_t.reshape()
    action = actor.getAction(s_t)
    s_t_old = s_t
    REWARD = r_t
    error = 0
    for t in xrange(0, 1000):
        if(norender == False):
            env.render()
        s_t, r_t, done, info = env.step(action)        
        
        REWARD += r_t
        # add to BATCH history {s_t_old, a_t, r_t, s_t}
        replay.add(s_t_old, action, r_t, s_t, done)
        s_t_old = s_t    
        # train NNet
        batch, ready = replay.getBatch(REPLAY_BUFFER_SIZE) 

        if(ready and do_train):
            error = actor.train(batch)
        
        action = actor.getAction(s_t)        
        if(done):
            break
    if(REWARD > SCORE_THOLD):
        n_episodes_thold += 1
        if(n_episodes_thold > 30):
            do_train=False
    else:
        n_episodes_thold = 0
    actor.decay_epsilon()
    print "Episode %5d ended with reward : %5d (NN error = %4.2f)"%(ep, REWARD, error)
    reward_eps[ep] = REWARD
    if(ep%100 == 0 and ep != 0):
        actor.save("./savedNN/ep_" + str(ep))

np.save(envname, reward_eps)
