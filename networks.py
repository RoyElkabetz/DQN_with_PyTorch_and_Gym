import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class LinearDeepNetwork(nn.Module):
	"""docstring for LinearDeepNetwork"""
	def __init__(self, input_dims, n_actions, lr):
		super().__init__()
		self.fc1 = nn.Linear(*input_dims, 128)
		self.fc2 = nn.Linear(128, n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		actions = self.fc2(x)
		return actions


class DQNetwork(nn.Module):
	def __init__(self, input_dims, n_actions, lr, name, chkpt_dir):
		super().__init__()
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

		self.conv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride=(4, 4))
		self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2))
		self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1))

		fc_input_dims = self.calculate_conv_output_dims(input_dims)
		self.fc1 = nn.Linear(fc_input_dims, 512)
		self.fc2 = nn.Linear(512, n_actions)

		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def calculate_conv_output_dims(self, input_dims):
		state = T.zeros(1, *input_dims)
		dims = self.conv1(state)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def forward(self, state):
		x = F.relu(self.conv1(state))
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = x.view(x.size()[0], -1)
		x = F.relu(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))


class DuelingDQNetwork(nn.Module):
	def __init__(self, input_dims, n_actions, lr, name, chkpt_dir):
		super().__init__()
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

		self.conv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride=(4, 4))
		self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2))
		self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1))

		fc_input_dims = self.calculate_conv_output_dims(input_dims)
		self.fc1 = nn.Linear(fc_input_dims, 512)
		self.V = nn.Linear(512, 1)
		self.A = nn.Linear(512, n_actions)

		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def calculate_conv_output_dims(self, input_dims):
		state = T.zeros(1, *input_dims)
		dims = self.conv1(state)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def forward(self, state):
		x = F.relu(self.conv1(state))
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = x.view(x.size()[0], -1)
		x = F.relu(x)
		x = self.fc1(x)
		x = F.relu(x)
		V = self.V(x)
		A = self.A(x)
		return V, A

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))

