import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
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
        self.__trained = False

    def _model(self):
        '''
        Build a new model
        '''
        #add model layers
        model = Sequential()
        model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(80,80, 1)))
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(80,80, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def _epsilon_greedy(self, state, epsilon):
        rand_num = np.random.random()
        if rand_num < epsilon:
            # Explore random move
            action = self.env.action_space.sample()
        else:
            #get best action
            action = np.argmax(self.model.predict(state))
        return action
    
    def train(self, step_limit=10000, epsilon=0.5, epsilon_decay=0.99, batch_size=64):
        self._model()
        self.__trained = True
        '''
                memory <- empty
				model <- untrained model
				for each episode:
					current-state = environment's initial state
					processed-state = process-state(state)
					for each step at playing the game or it timeouts:
						action <- make epsilon-greedy decision based on the current state
						get reward, next state, is_terminal from the actions on the environment
						store this transition (s_{t}, action, s_{t + 1), is_terminal) into memory
						batch <- sample a random batch (might consider batch size here)
						rewards = empty
						for each sample in the batch:
							label each sample's reward with
							reward if terminal state
							reward = reward + gamma * model's prediction on next state
							add reward to rewards
						train the model on the batch's states and rewards
        '''
        for _ in range(self.num_eps):
            curr_view = self.env.reset()
            curr_pview = self._processed_frame(curr_view) #to be stored
            
            for _ in range(step_limit):
                action = self._epsilon_greedy(curr_pview, epsilon)
                next_view, reward, done, _ = self.env.step(action)
                next_pview = self._processed_frame(next_view)
                self.mem.append((curr_pview, reward, done, next_pview))
                curr_pview = next_pview
                if len(self.mem) > batch_size:
                    pviews, rewards = self.training_data(batch_size)
                    self.model.train_on_batch(pviews, rewards)
                if done:
                    break
                if epsilon <= 0.001:
                    epsilon *= epsilon_decay
                
    def training_data(self, batch_size):
        batch = random.sample(self.mem, batch_size)
        pviews = []
        rewards = []
        for pview, reward, done, next_pview in batch:
            pviews.append(pview)
            if done:
                rewards.append(rewards)
            else:
                rewards.append(rewards + self.gamma*np.amax(self.model.predict(next_pview)))
        return np.array(pviews), np.array(rewards)

    def play(self, step_limit=10000):
        '''
        Run one episode
        Arguments: step_limit: How many steps it can play before something we dropout
        '''
        view = self._processed_frame(self.env.reset())
        previous_view = None
        for _ in range(step_limit):
            continue
        pass
            

    def _processed_frame(self, view : np.ndarray):
        '''
        Using karpathy's impl
        '''
        view = view[35:195] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        view = view[::2,::2,0] # downsample by factor of 2.
        view[view == 144] = 0 # erase background (background type 1)
        view[view == 109] = 0 # erase background (background type 2)
        view[view != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        view = np.expand_dims(view.astype(np.float), axis=-1) # ravel flattens an array and collapses it into a column vector
        return np.resize(view, (1, 80, 80, 1))

if __name__ == '__main__':
    env = gym.make('Pong-v0')
    view = env.reset()
    def processed_frame(view : np.ndarray):
        view = view[35:195] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        view = view[::2,::2,0] # downsample by factor of 2.
        view[view == 144] = 0 # erase background (background type 1)
        view[view == 109] = 0 # erase background (background type 2)
        view[view != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return view.astype(np.float) # ravel flattens an array and collapses it into a column vector
    print(processed_frame(view).shape)
    agent = Agent(env, env.action_space.n)
    agent.train()

