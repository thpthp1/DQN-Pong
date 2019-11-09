import gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten
import numpy as np
import scipy
from collections import deque
import matplotlib.pyplot as plt
import random
from numpy.random import choice
import sys
import time
from gym import wrappers
class Agent(object):

    def __init__(self, env, actions, action_map, gamma=0.9, num_eps=10000, mem_len=2000):
        self.env = env
        self.model = None
        self.trainer = None
        self.gamma = gamma
        self.num_eps = num_eps
        self.mem = deque(maxlen=mem_len)
        self.actions = actions
        self.action_map = action_map
        self.__trained = False
        
    def m_loss(self, episode_reward):
        def loss(y_true,y_pred):
            # feed in y_true as actual action taken 
            # if actual action was up, we feed 1 as y_true and otherwise 0
            # y_pred is the network output(probablity of taking up action)
            # note that we dont feed y_pred to network. keras computes it

            # first we clip y_pred between some values because log(0) and log(1) are undefined
            tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(y_pred)
            # we calculate log of probablity. y_pred is the probablity of taking up action
            # note that y_true is 1 when we actually chose up, and 0 when we chose down
            # this is probably similar to cross enthropy formula in keras, but here we write it manually to multiply it by the reward value
            tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(tmp_pred)
            # multiply log of policy by reward
            policy_loss=keras.layers.Multiply()([tmp_loss,episode_reward])
            return policy_loss
        return loss

    def _model(self):
        '''
        Build a new model
        '''
        inputs = keras.layers.Input(shape=(80*80, ))
        flattened_layer = keras.layers.Flatten()(inputs)
        full_connect_1 = Dense(units=200,activation='relu',use_bias=False,)(flattened_layer)
        sigmoid_output = Dense(1,activation='sigmoid',use_bias=False)(full_connect_1)
        policy_network_model = keras.models.Model(inputs=inputs,outputs=sigmoid_output)
        policy_network_model.summary()
                               
        episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')
        policy_network_train = keras.models.Model(inputs=[inputs,episode_reward],outputs=sigmoid_output)

        my_optimizer = keras.optimizers.RMSprop(lr=0.0001)
        policy_network_train.compile(optimizer=my_optimizer,loss=self.m_loss(episode_reward))
        self.model = policy_network_model
        self.trainer = policy_network_train
        return self.model

    def _epsilon_greedy(self, state, epsilon):
        rand_num = np.random.uniform()
        state = np.expand_dims(state.astype(np.float), axis=0)
#         if rand_num < epsilon:
#             # Explore random move
#             action = choice(range(self.actions), 1)[0]
#         else:
#             #get best action
#             action = np.argmax(self.model.predict(state)[0])
#         print(self.model.predict(state))
        action = 1 if self.model.predict(state)[0] > rand_num else 0
        return action
    
    def train(self, step_limit=2000, epsilon=0.9, epsilon_decay=0.95, batch_size=2):
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
        transitions = []
        rewards_eps = []
        rewards = []
        for _ in range(self.num_eps):
            pre_pview = None
            curr_view = self.env.reset()
            curr_pview = self._processed_frame(curr_view) #to be stored         
            
            for _ in range(step_limit):
                transition = curr_pview - pre_pview if pre_pview is not None else np.zeros_like(curr_pview)
                action = self._epsilon_greedy(transition, epsilon)
                rewards_eps.append(float(action))
                transitions.append(transition)
                next_view, reward, done, _ = self.env.step(self.action_map[action])
                next_pview = self._processed_frame(next_view)
                rewards.append(reward)
                pre_pview = curr_pview
                curr_pview = next_pview
#                 if len(self.mem) > batch_size:
#                     pviews, rewards = self.training_data(batch_size)
#                     loss = self.model.train_on_batch(pviews, rewards)
                if done:
                    print('Finished full game')
                    break
#                 if epsilon >= 0.01:
#                     epsilon *= epsilon_decay
            if len(transitions) % batch_size == 0:
              print(np.array(transitions).shape)
              print(np.sum(rewards_eps))
              self.train.fit([np.array(transitions), self.discount_reward(rewards, self.gamma)], np.array(rewards_eps))
              transitions, rewards_eps, rewards = [], [], []
            print('Finished one episode with total reward ', np.sum(rewards))
                
    def training_data(self, batch_size):
        batch = random.sample(self.mem, batch_size)
        pviews = []
        rewards = []
        for pview, reward, done, action, next_pview in batch:
            pviews.append(pview)
            next_pview = self._to_model_data(next_pview)
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_pview)[0])
            reward_model = self.model.predict(self._to_model_data(pview))
            reward_model[0][action] = target
            rewards.append(reward_model[0])
        rewards_normed = np.array(rewards)
#         rewards_normed -= np.mean(rewards)
#         rewards_normed /= np.std(rewards)
        return np.array(pviews), np.array(rewards)

    def discount_rewards(self, r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        r = np.array(r)
        discounted_r = np.zeros_like(r)
        running_add = 0
        # we go from last reward to first one so we don't have to do exponentiations
        for t in reversed(range(0, r.size)):
          if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
          running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
          discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r) #normalizing the result
        discounted_r /= np.std(discounted_r) #idem
        return discounted_r

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
        # view = np.expand_dims(view.astype(np.float), axis=0) # ravel flattens an array and collapses it into a column vector
        # return np.resize(view, (1, 80, 80, 1))
        return np.ravel(view)
      
    def _to_model_data(self, view):
        return np.expand_dims(view.astype(np.float), axis=0)

if __name__ == '__main__':
    env = gym.make('Pong-v0')
    view = env.reset()
    agent = Agent(env, actions=2, action_map={1:2, 0:3}, num_eps=100)
    agent.train()
    env.close()
    env = gym.make('Pong-v0')
    env = wrappers.Monitor(env, "./gym-results", force=True)
    view = env.reset()
    for _ in range(1000):
        action = agent._epsilon_greedy(agent._processed_frame(view), 0.5)
        observation, reward, done, info = env.step(agent.action_map[action])
        if done: break
    env.close()
    

