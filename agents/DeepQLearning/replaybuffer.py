# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:04:10 2016

@author: fcaldas
"""
import numpy as np

class ReplayBuffer():
    rotated = False    
    
    def __init__(self, size):
        self.buffer = np.zeros([size, 11])
        self.nElements = 0
        self.c = 0
        self.limit = size
    
    def add(self, s_t, a_t, r_t, s_tt, done):
        if(self.nElements < self.limit):
            self.buffer[self.nElements, 0:4] = s_t
            self.buffer[self.nElements, 4] = a_t
            self.buffer[self.nElements, 5] = r_t
            self.buffer[self.nElements, 6:10] = s_tt
            self.buffer[self.nElements, 10] = done
            self.nElements += 1
            self.c+=1
        else:
            self.buffer[self.c%self.limit, 0:4] = s_t
            self.buffer[self.c%self.limit, 4] = a_t
            self.buffer[self.c%self.limit, 5] = r_t
            self.buffer[self.c%self.limit, 6:10] = s_tt
            self.buffer[self.c%self.limit, 10] = done
            self.c+=1
    
    def getBatch(self, size):
        if(size >= self.nElements):
            return [self.buffer[0:self.nElements, :], False]
        else:
            idxes = np.random.randint(0,self.nElements,size=size)
            return [self.buffer[idxes], True]
            
            
            
if(__name__=="__main__"):
    s = np.array([1,2,3,4])
    rb = ReplayBuffer(10)
    rb.add(s,1,2,s,0)
    rb.add(s,1,2,s,0)
    rb.add(s,1,2,s,0)
    rb.add(s,1,2,s,0)
    rb.add(s,1,2,s,0)
    print rb.getBatch(2)
