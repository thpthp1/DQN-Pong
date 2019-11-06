import gym
import tensorflow as tf
import keras
import numpy as np
import scipy
from collections import deque
import matplotlib.pyplot as plt
import random

class Agent(object):

    def __init__(self, env, actions, gamma=0.75, num_eps=10000, mem_len=2000):
        self.env = env
        self.model = None
        self.gamma = gamma
        self.num_eps = num_eps
        self.mem = deque(maxlen=mem_len)
        self.actions = actions

    def _model(self):
        '''
        Build a new model
        '''
        pass

    def _epsilon_greedy(self, epsilon, state):
        rand_num = np.random.random()
        if rand_num < epsilon:
            # Explore random move
            action = self.env.action_space.sample()
        else:
            #get best action
            action = np.argmax(self.model.predict(state))
        return action
    
    def train(self):
        pass

    def replay(self):
        pass

    def play(self):
        pass


def processed_frame(view : np.ndarray):
    view = view[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    view = view[::2,::2,0] # downsample by factor of 2.
    view[view == 144] = 0 # erase background (background type 1)
    view[view == 109] = 0 # erase background (background type 2)
    view[view != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return view.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

if __name__ == '__main__':
    env = gym.make('Pong-v0')
    view = env.reset()
    

