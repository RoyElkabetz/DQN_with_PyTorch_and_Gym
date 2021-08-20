import numpy as np
from dqn_networks import LinearDeepNetwork, DeepQNetwork, DuelingDeepQNetwork
import torch as T
from replay_memory import ReplayBuffer


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
		loss = self.Q.loss(q_pred, q_target).to(self.Q.device)
		loss.backward()
		self.Q.optimizer.step()
		self.decerement_epsilon()


class DeepQNAgent:
	def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
				 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
				 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):

		self.gamma = gamma
		self.epsilon = epsilon
		self.lr = lr
		self.n_actions = n_actions
		self.input_dims = input_dims
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.eps_min = eps_min
		self.eps_dec = eps_dec
		self.replace_target_count = replace
		self.algo = algo
		self.env_name = env_name
		self.chkpt_dir = chkpt_dir
		self.action_space = [i for i in range(self.n_actions)]
		self.learn_step_counter = 0

		self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

		self.q_eval = DeepQNetwork(self.input_dims, self.n_actions, self.lr,
									name=self.env_name + '_' + self.algo + '_q_eval',
									chkpt_dir=self.chkpt_dir)
		self.q_next = DeepQNetwork(self.input_dims, self.n_actions, self.lr,
									name=self.env_name + '_' + self.algo + '_q_next',
									chkpt_dir=self.chkpt_dir)

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
			actions = self.q_eval.forward(state)
			action = T.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)

		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def sample_memory(self):
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		states = T.tensor(state).to(self.q_eval.device)
		actions = T.tensor(action).to(self.q_eval.device)
		rewards = T.tensor(reward).to(self.q_eval.device)
		states_ = T.tensor(new_state).to(self.q_eval.device)
		dones = T.tensor(done).to(self.q_eval.device)

		return states, actions, rewards, states_, dones

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_target_count == 0:
			self.q_next.load_state_dict(self.q_eval.state_dict())

	def save_models(self):
		self.q_eval.save_checkpoint()
		self.q_next.save_checkpoint()

	def load_models(self):
		self.q_eval.load_checkpoint()
		self.q_next.load_checkpoint()

	def learn(self):
		if self.learn_step_counter < self.batch_size:
			self.learn_step_counter += 1
			return

		self.q_eval.optimizer.zero_grad()
		self.replace_target_network()
		states, actions, rewards, states_, dones = self.sample_memory()

		# Deep Q learning update rule
		indices = np.arange(self.batch_size)
		q_pred = self.q_eval(states)[indices, actions]
		q_next = self.q_next(states_).max(dim=1)[0]

		q_next[dones] = 0.0
		q_target = rewards + self.gamma * q_next
		loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.decrement_epsilon()


class DoubleDeepQNAgent:
	def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
				 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
				 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):

		self.gamma = gamma
		self.epsilon = epsilon
		self.lr = lr
		self.n_actions = n_actions
		self.input_dims = input_dims
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.eps_min = eps_min
		self.eps_dec = eps_dec
		self.replace_target_count = replace
		self.algo = algo
		self.env_name = env_name
		self.chkpt_dir = chkpt_dir
		self.action_space = [i for i in range(self.n_actions)]
		self.learn_step_counter = 0

		self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

		self.q_eval = DeepQNetwork(self.input_dims, self.n_actions, self.lr,
									name=self.env_name + '_' + self.algo + '_q_eval',
									chkpt_dir=self.chkpt_dir)
		self.q_next = DeepQNetwork(self.input_dims, self.n_actions, self.lr,
									name=self.env_name + '_' + self.algo + '_q_next',
									chkpt_dir=self.chkpt_dir)

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
			actions = self.q_eval.forward(state)
			action = T.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)

		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def sample_memory(self):
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		states = T.tensor(state).to(self.q_eval.device)
		actions = T.tensor(action).to(self.q_eval.device)
		rewards = T.tensor(reward).to(self.q_eval.device)
		states_ = T.tensor(new_state).to(self.q_eval.device)
		dones = T.tensor(done).to(self.q_eval.device)

		return states, actions, rewards, states_, dones

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_target_count == 0:
			self.q_next.load_state_dict(self.q_eval.state_dict())

	def save_models(self):
		self.q_eval.save_checkpoint()
		self.q_next.save_checkpoint()

	def load_models(self):
		self.q_eval.load_checkpoint()
		self.q_next.load_checkpoint()

	def learn(self):
		if self.learn_step_counter < self.batch_size:
			self.learn_step_counter += 1
			return

		self.q_eval.optimizer.zero_grad()
		self.replace_target_network()
		states, actions, rewards, states_, dones = self.sample_memory()

		# Double Deep Q learning update rule
		indices = np.arange(self.batch_size)
		q_pred = self.q_eval(states)[indices, actions]
		max_actions = T.argmax(self.q_eval(states_), dim=1)
		q_next = self.q_next(states_)[indices, max_actions]

		q_next[dones] = 0.0
		q_target = rewards + self.gamma * q_next
		loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.decrement_epsilon()


class DuelingDeepQNAgent:
	def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
				 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
				 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):

		self.gamma = gamma
		self.epsilon = epsilon
		self.lr = lr
		self.n_actions = n_actions
		self.input_dims = input_dims
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.eps_min = eps_min
		self.eps_dec = eps_dec
		self.replace_target_count = replace
		self.algo = algo
		self.env_name = env_name
		self.chkpt_dir = chkpt_dir
		self.action_space = [i for i in range(self.n_actions)]
		self.learn_step_counter = 0

		self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

		self.q_eval = DuelingDeepQNetwork(self.input_dims, self.n_actions, self.lr,
										name=self.env_name + '_' + self.algo + '_q_eval',
										chkpt_dir=self.chkpt_dir)
		self.q_next = DuelingDeepQNetwork(self.input_dims, self.n_actions, self.lr,
										name=self.env_name + '_' + self.algo + '_q_next',
										chkpt_dir=self.chkpt_dir)

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
			_, advantage = self.q_eval.forward(state)
			action = T.argmax(advantage).item()
		else:
			action = np.random.choice(self.action_space)

		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def sample_memory(self):
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		states = T.tensor(state).to(self.q_eval.device)
		actions = T.tensor(action).to(self.q_eval.device)
		rewards = T.tensor(reward).to(self.q_eval.device)
		states_ = T.tensor(new_state).to(self.q_eval.device)
		dones = T.tensor(done).to(self.q_eval.device)

		return states, actions, rewards, states_, dones

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_target_count == 0:
			self.q_next.load_state_dict(self.q_eval.state_dict())

	def save_models(self):
		self.q_eval.save_checkpoint()
		self.q_next.save_checkpoint()

	def load_models(self):
		self.q_eval.load_checkpoint()
		self.q_next.load_checkpoint()

	def learn(self):
		if self.learn_step_counter < self.batch_size:
			self.learn_step_counter += 1
			return

		self.q_eval.optimizer.zero_grad()
		self.replace_target_network()
		states, actions, rewards, states_, dones = self.sample_memory()

		# Dueling Deep Q learning update rule
		indices = np.arange(self.batch_size)
		V_s, A_s = self.q_eval.forward(states)
		V_s_, A_s_ = self.q_next.forward(states_)
		q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
		q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

		q_next[dones] = 0.0
		q_target = rewards + self.gamma * q_next

		loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.decrement_epsilon()


class DuelingDoubleDeepQNAgent:
	def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
				mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
				replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):

		self.gamma = gamma
		self.epsilon = epsilon
		self.lr = lr
		self.n_actions = n_actions
		self.input_dims = input_dims
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.eps_min = eps_min
		self.eps_dec = eps_dec
		self.replace_target_count = replace
		self.algo = algo
		self.env_name = env_name
		self.chkpt_dir = chkpt_dir
		self.action_space = [i for i in range(self.n_actions)]
		self.learn_step_counter = 0

		self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

		self.q_eval = DeepQNetwork(self.input_dims, self.n_actions, self.lr,
									name=self.env_name + '_' + self.algo + '_q_eval',
									chkpt_dir=self.chkpt_dir)
		self.q_next = DeepQNetwork(self.input_dims, self.n_actions, self.lr,
									name=self.env_name + '_' + self.algo + '_q_next',
									chkpt_dir=self.chkpt_dir)

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
			actions = self.q_eval.forward(state)
			action = T.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)

		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def sample_memory(self):
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		states = T.tensor(state).to(self.q_eval.device)
		actions = T.tensor(action).to(self.q_eval.device)
		rewards = T.tensor(reward).to(self.q_eval.device)
		states_ = T.tensor(new_state).to(self.q_eval.device)
		dones = T.tensor(done).to(self.q_eval.device)

		return states, actions, rewards, states_, dones

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_target_count == 0:
			self.q_next.load_state_dict(self.q_eval.state_dict())

	def save_models(self):
		self.q_eval.save_checkpoint()
		self.q_next.save_checkpoint()

	def load_models(self):
		self.q_eval.load_checkpoint()
		self.q_next.load_checkpoint()

	def learn(self):
		if self.learn_step_counter < self.batch_size:
			self.learn_step_counter += 1
			return

		self.q_eval.optimizer.zero_grad()
		self.replace_target_network()
		states, actions, rewards, states_, dones = self.sample_memory()

		# Dueling Double Deep Q learning update rule
		indices = np.arange(self.batch_size)
		V_s, A_s = self.q_eval.forward(states)
		V_s_, A_s_ = self.q_next.forward(states_)
		V_s_eval, A_s_eval = self.q_eval.forward(states_)

		q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepsim=True)))[indices, actions]
		q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepsim=True)))
		q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

		max_actions = T.argmax(q_eval, dim=1)

		q_next[dones] = 0.0
		q_target = rewards + self.gamma * q_next[indices, max_actions]
		loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.decrement_epsilon()
