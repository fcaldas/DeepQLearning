# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:05:06 2016

@author: fcaldas
"""
import numpy as np

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

class Tiles():
    
    def __init__(self, nafterComma, outputDim):
        self.cmac = {}
        self.discretization = nafterComma
        self.outputDim = outputDim
    
    def getIdx(self, state):
        disc = self.discretization
        idx = np.zeros(state.shape)
        stridx = ""
        for i in range(0, disc.shape[0]):
            idx[i] = np.round(np.abs(state)[i]/disc[i])
            stridx = stridx + str(np.sign(idx[i])) + str(idx[i]) + ";"
        print stridx
        return stridx
    
    def setV(self, state, v):
        idx = self.getIdx(state)
        self.cmac[idx] = v
        print len(self.cmac.keys())
    
    def get(self, state):
        idx = self.getIdx(state)
        if(idx not in self.cmac):
            return np.zeros(self.outputDim)
        else:
            return self.cmac[idx]