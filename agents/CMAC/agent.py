# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:39:33 2016

@author: fcaldas
"""

from Tile import *
import numpy as np


class CMACAgent():
    maxspace = np.array([0.15, 2, np.pi, np.pi*2])
    epsilon_decay = 0.99
    epsilon = 0.10
    gamma = 0.98
    A_1 = None
    S_1 = None
    alpha = 0.7
    
    def __init__(self, actions=np.linspace(-0.7, 0.7, num=30)):
        self.tile = Tiles(np.array([0.05, 0.5, np.pi/20 , np.pi/10]),
                          actions.shape[0])
        self.actions = actions
    
    def epsilon_action(self):
        return np.random.choice(self.actions)
        
    def greedy_action(self, S):
        Qvs = np.zeros(self.actions.shape[0])
        Qvs = self.tile.get(S)
        idx = np.random.choice(np.where(np.max(Qvs) == Qvs)[0])
        return self.actions[idx]
        
    
    def getAction(self, S, r=None, done=False):
        isepsilon = False
        alpha = self.alpha
        if(self.epsilon >= np.random.rand()):
            A = self.epsilon_action()
            isepsilon = True
        else:
            A = self.greedy_action(S)
        
        S_1 = self.S_1
        A_1 = self.A_1
        if(r is not None and not(isepsilon) and S_1 is not None):
            #train the CMAC
            # Q(s,a) = (1-alpha)*Q(s,a) + alpha(r + \gamma Q(s+, a+))
            idx_A = np.where(self.actions == A)[0][0]
            idx_A_1 = np.where(self.actions == A_1)[0][0]
            Q_S = self.tile.get(S)
            Q_S_1 = self.tile.get(S_1)
            if(not done):
                Q_target = (1-alpha) * Q_S_1[idx_A_1]\
                          + alpha * ( r + self.gamma * Q_S[idx_A])
            else:
                Q_target = r
            Q_S_1[idx_A_1] = Q_target
            self.tile.setV(S_1, Q_S_1)
        self.S_1 = S
        self.A_1 = A
        return A
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay