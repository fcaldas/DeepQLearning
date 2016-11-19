# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:10:57 2016

@author: fcaldas
"""
import numpy as np
from neuralnet import *


class DNN():
    
    
    def __init__(self, batch_size, actions=np.linspace(-1,1,num=50)):
        
        self.NN1 = NNet(batch_size, actions=actions)
        self.NN2 = NNet(batch_size, actions=actions)
        
    def getAction(self, X):
        if(np.random.rand() < self.NN1.epsilon):
            v = np.random.choice(self.NN1.actions_discrete)
        else:
            #calculate the average of both NNs                
            Q1 = self.NN1.getGreedyValues(X)
            #Q2 = self.NN2.getGreedyValues(X)
            Qa = np.mean([Q1], axis=0)
            idx = np.random.choice(np.where(np.max(Qa) == Qa)[0])
            v = self.NN1.actions_discrete[idx]
        return v
    
    def copyNet(self):
        self.NN2.weights = self.NN1.weights
        self.NN2.biases = self.NN1.biases
    
    def decay_epsilon(self):
        self.NN1.decay_epsilon()
        self.NN2.decay_epsilon()
        
    def train(self, batch1):
        # calculate greedy actions using NN1
        r = batch1[:, 5]
        done = batch1[:, 10]
        
        greedy_actions = np.zeros([batch1.shape[0],1])
        splus = batch1[:, 6:10]
        for i in range(0, batch1.shape[0]):
            greedy_actions[i] = self.NN1.getGreedy(splus[i, :])
        evaluate = np.zeros([batch1.shape[0], 5])
        evaluate[:, 0:4] = batch1[:, 6:10]
        evaluate[:, 4] = greedy_actions.reshape([batch1.shape[0]])
        Qmax = self.NN2.evaluate(evaluate)
        
        # target = r_t + max Q(s_tt, )+
        target = batch1[:, 5].reshape([batch1.shape[0], 1]) + self.NN1.discount_rate * Qmax

        final_states = np.where(done == 1)[0]
        target[final_states] = r[final_states].reshape([final_states.shape[0], 1])
        # train the first NN with the first batch
        target = target.reshape([target.shape[0], 1])
        e1 = self.NN1.train(batch1[:, 0:5], target)

        return e1