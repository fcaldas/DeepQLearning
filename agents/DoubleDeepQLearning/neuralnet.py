# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:50:28 2016

@author: fcaldas

Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""

import tensorflow as tf
import numpy as np

class NNet():
    nInputs = 5
    layers = 3
    nOutputs = 1
    
    actions_discrete = np.linspace(-1,1,num=50)
    nActions = 50
    discount_rate = 0.99
    epsilon = 0.25
    epsilon_decay = 0.998
   
    
    weights = {
        'h1': tf.Variable(tf.random_uniform([5, 250], minval=-0.05, maxval=0.05)),
        'h2': tf.Variable(tf.random_uniform([250, 250], minval=-0.05, maxval=0.05)),
        #'h3': tf.Variable(tf.random_uniform([250, 250], minval=-0.05, maxval=0.05)),
        'out': tf.Variable(tf.random_uniform([250, 1], minval=-0.05, maxval=0.05))
    }

    biases = {
        'b1': tf.Variable(tf.random_uniform([250], minval=-0.05, maxval=0.05)),
        'b2': tf.Variable(tf.random_uniform([250], minval=-0.05, maxval=0.05)),
        #'b3': tf.Variable(tf.random_uniform([250], minval=-0.05, maxval=0.05)),
        'out': tf.Variable(tf.random_uniform([1], minval=-0.05, maxval=0.05))
    }

    def createNN(self):
        self.state = tf.placeholder(tf.float32, shape=[None, self.nInputs])
        # Hidden layer with RELU activation
        layer_1 = (tf.matmul(self.state, self.weights['h1']))
        layer_1 = tf.nn.relu(layer_1) #use log(1+e^x)
        # Hidden layer with RELU activation
        layer_2 = (tf.matmul(layer_1, self.weights['h2']))
        layer_2 = tf.nn.relu(layer_2)
        # One extra hidden layer
        #layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3'])
        #layer_3 = tf.nn.relu(layer_3)       
        # Output layer with linear activation
        self.y = tf.placeholder("float32", [None, 1], name="OUTPUT")        
        self.out_layer = tf.identity(tf.matmul(layer_2, self.weights['out']) + self.biases['out'])
        
        
    def __init__(self, batch_size, actions=np.linspace(-1,1,num=50)):
        self.nActions = actions.shape[0]
        self.actions_discrete=actions
        #save space for these so we won't need to realocate them every iter
        print "[+] Initializing neural network"
        self.batchHolder = np.zeros([batch_size, 5])
        self.batch_size = batch_size
        
        #initilize Neural Net using tensorflow
        self.createNN();    
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.out_layer, self.y))
        self.target = tf.placeholder("float32", [None, 1], name="TARGET")
        #self.cost = tf.nn.l2_loss(self.out_layer - self.target)
        self.cost = tf.reduce_mean(tf.sqrt(1 + tf.square(tf.sub(self.out_layer, self.target))) - 1)

        #self.cost = tf.reduce_mean(tf.square(self.out_layer - self.target))
        self.NN_learning_rate = 0.00025  #tf.train.exponential_decay(0.01, 
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.NN_learning_rate).minimize(self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.NN_learning_rate).minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.NN_learning_rate).minimize(self.cost)
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(self.init)


    def save(self, filename):
        self.saver.save(self.session, filename)

    def load(self, filename):
        self.saver.restore(self.session, filename)

    def evaluate(self, s):
        return self.session.run(self.out_layer, feed_dict={
            self.state: s
        })
        
        
    # For a set of states S', find a' that maximizes Q(S',a')
    # Returns max(a') Q(S',a')
    def _getMaxOutput(self, Xl):
        #Q can be negative therefore we use -inf
        batchMax = np.ones([self.batch_size]) * (- np.inf)
        self.batchHolder[:,0:4] = Xl
        for j in self.actions_discrete:
            # everyone in batch does the same action
            self.batchHolder[:,4] = j
            q = self.evaluate(self.batchHolder)
            q = q.reshape(q.shape[0])
            #choose max between batchMax and Q()
            batchMax = np.max([batchMax, q], axis=0)
        return batchMax

    def _getMultipleGreedy(self, Xl):
        actions = np.zeros(Xl.shape[0])
        for i in xrange(0, Xl.shape[0]):
            actions[i] = self.getGreedy(Xl[i, :])
        return actions
    
    # Return greedy action for state S
    # return argmax(a) Q(S, a) 
    def getGreedy(self, X):
        possibleStates = np.zeros([self.nActions, 5])
        possibleStates[:, 0:4] = X
        possibleStates[:, 4] = self.actions_discrete
        Q = self.evaluate(possibleStates)
        # print np.max(Q)
        # print np.where(np.max(Q) == Q)
        idx = np.random.choice(np.where(np.max(Q) == Q)[0])
        return self.actions_discrete[idx]

    # epsilon policy of our actor        
    def getEpsilon(self):
        return np.random.choice(self.actions_discrete)
    
    def getGreedyValues(self, X):
        possibleStates = np.zeros([self.nActions, 5])
        possibleStates[:, 0:4] = X
        possibleStates[:, 4] = self.actions_discrete
        Q = self.evaluate(possibleStates)
        return Q        
    
    def getAction(self, S):
        if(np.random.rand() < self.epsilon):
            return self.getEpsilon()
        else:
            return self.getGreedy(S)
            
    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        
    # train(): [s_t, a_t, r_t, s_t', done]
    # receives a batch with dimension [n, 11]     
    def train(self, batch, target):
        # calculate target
        X = batch[:, 0:5]

        _, c = self.session.run([self.optimizer, self.cost], feed_dict={self.state: X,
                                                                        self.target: target})
        return c
