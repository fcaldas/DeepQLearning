# -*- coding: utf-8 -*-

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import pi, sin, cos
from os import path

logger = logging.getLogger(__name__)

class InvPoliPoleEnv(gym.Env):
    
    M = 0.6242;
    m = 0.1722;
    b = 5.;
    J = 6.0306e-04;
    g = 9.8;
    l = 0.41/2;
    
    # motor parameters
    n = 5.9;
    r = 0.0283/2;
    Rm = 1.4;
    Ke = 12./(2 * pi * n * 1000/(60));
    Kt = 1.5/(n * 15.5);

    X = 0.; # car position
    X_d = 0.; # car speed
    P = 0.; # pendulum angle (in rads 0 = up)
    P_d = 0.; # angular speed
    
    dt = 0.02
    lastAction = None
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.25

        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max, np.pi, np.finfo(np.float32).max])

        # continuous action space
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.lastAction = action
#        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        self.X = x
        self.X_d = x_dot
        self.P = theta
        self.P_d = theta_dot
        PWM=action; # no point in multiplying and dividing
        X_dd = -(self.Rm*sin(self.P)*self.l**3*self.m**2*self.P_d**2*self.r**2 - self.Rm*self.g*cos(self.P)*\
                sin(self.P)*self.l**2*self.m**2*self.r**2 + self.Rm*self.b*self.X_d*self.l**2*self.m\
                *self.r**2-12*self.Kt*PWM*self.n*self.l**2*self.m*self.r + self.Ke*self.Kt*self.X_d*self.l**2\
                *self.m + self.J*self.Rm*sin(self.P)*self.l*self.m*self.P_d**2*self.r**2 + self.J*\
                self.Rm*self.b*self.X_d*self.r**2-12*self.J*self.Kt*PWM*self.n*self.r+\
                self.J*self.Ke*self.Kt*self.X_d)/(self.Rm*self.r**2*(self.J*self.m + self.J*self.M\
                + self.l**2*self.m**2 - self.l**2*self.m**2*cos(self.P)**2 + self.M*self.l**2*self.m));
        
        P_dd = -(self.l*self.m*(self.Ke*self.Kt*self.X_d*cos(self.P) - self.M*self.Rm*self.g*self.r**2\
                *sin(self.P) + self.Rm*self.b*self.r**2*self.X_d*cos(self.P) - self.Rm*self.g*self.m*self.r**2*\
                sin(self.P)-12*self.Kt*PWM*self.n*self.r*cos(self.P)+self.Rm*self.l*self.m*self.P_d**2*self.r**2*\
                cos(self.P)*sin(self.P)))/(self.Rm*self.r**2*(self.J*self.m + self.J*self.M +\
                self.l**2*self.m**2 - self.l**2*self.m**2*cos(self.P)**2 + self.M*self.l**2*self.m));
                
        x_dot = x_dot + self.dt * X_dd
        x  = x + self.dt * x_dot        
        theta_dot = theta_dot + self.dt * P_dd
        theta = theta + self.dt * theta_dot
        
        # limit theta in [-pi, pi]
        theta = theta % (2 * pi)
        if(theta > pi):
            theta = theta - (2 * pi )
        elif(theta < -pi):
            theta = (2 * pi) - theta
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        done =  x < -self.x_threshold \
                or x > self.x_threshold or (np.abs(theta_dot) > 2*np.pi and np.abs(theta) < 0.05)
        
        done = bool(done)

        if not done:
            reward = np.abs(np.cos(theta/2)) / (0.8 + np.abs(np.sin(theta/2))) * 4
        else:
            reward = -10.
            
        return self.state, reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4))
        self.state[2] += pi 
        self.steps_beyond_done = None
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*5
        scale = screen_width/world_width
        carty = 190 # TOP OF CART
        polewidth = 10.0
        polelen = 150 * 1.
        cartwidth = 50.0
        cartheight = 30.0
        
        x = self.state
        cartx = x[0]*scale+screen_width/2.0
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering    
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)

        
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
