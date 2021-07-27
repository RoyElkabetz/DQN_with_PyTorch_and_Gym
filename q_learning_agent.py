import numpy as np


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
		