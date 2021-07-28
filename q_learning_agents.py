import numpy as np
from dqn_networks import LinearDeepNetwork
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class Agent:
	"""An agent for Q learning"""
	def __init__(self, lr, gamma, eps_start, eps_end, eps_dec, 
				n_states, n_actions):
		self.lr = lr
		self.gamma = gamma
		self.epsilon = eps_start
		self.eps_min = eps_end
		self.eps_dec = eps_dec
		self.n_states = n_states
		self.n_actions = n_actions
		self.Q = {}
		self.set_Q()

	def set_Q(self):
		for state in range(self.n_states):
			for action in range(self.n_actions):
				self.Q[(state, action)] = 0.0

	def choose_action(self, state):
		if np.random.rand() < self.epsilon:
			return np.random.choice(np.arange(self.n_actions))
		else:
			a_max = np.argmax([self.Q[(state, a)] for a in range(self.n_actions)])
			return a_max

	def learn(self, state, action, reward, next_state):
		a_max = np.argmax([self.Q[(next_state, a)] for a in range(self.n_actions)])
		self.Q[(state, action)] += self.lr * (reward + self.gamma * self.Q[(next_state, a_max)] - self.Q[(state, action)])
		self.decrease_epsilon()

	def decrease_epsilon(self):
		if self.epsilon > self.eps_min:
			self.epsilon *= self.eps_dec
		else: 
			pass
		

class DQNAgent:
	"""docstring for DQNAgent"""
	def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99, epsilon=1., 
				eps_min=0.01, eps_dec=1e-5):
		super().__init__()
		self.input_dims = input_dims
		self.lr = lr
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_min = eps_min
		self.eps_dec = eps_dec
		self.n_actions = n_actions
		self.action_space = [i for i in range(self.n_actions)]
		self.Q = LinearDeepNetwork(self.input_dims, self.n_actions, self.lr)

	def choose_action(self, observation):
		if np.random.rand() < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			state = T.tensor(observation, dtype=T.float).to(self.Q.device)
			actions = self.Q.forward(state)
			action = T.argmax(actions).item()
		return action

	def decerement_epsilon(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def learn(self, state, action, reward, next_state):
		self.Q.optimizer.zero_grad()

		states = T.tensor(state, dtype=T.float).to(self.Q.device)
		actions = T.tensor(action).to(self.Q.device)
		rewards = T.tensor(reward).to(self.Q.device)
		next_states = T.tensor(next_state, dtype=T.float).to(self.Q.device)

		
		q_pred = self.Q.forward(states)[actions]
		q_next = self.Q.forward(next_states).max()
		q_target = rewards + self.gamma * q_next
		loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
		loss.backward()
		self.Q.optimizer.step()
		self.decerement_epsilon()



		










		