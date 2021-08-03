import collections
import cv2

import numpy as np
import matplotlib.pyplot as plt
import gym


def plot_learning_curve(x, scores, epsilons, filename):
	fig = plt.figure()
	ax = fig.add_subplot(111, label='1')
	ax2 = fig.add_subplot(111, label='2', frame_on=False)

	ax.plot(x, epsilons, color='C0')
	ax.set_xlabel('Training Steps', color='C0')
	ax.set_ylabel('Epsilon', color='C0')
	ax.tick_params(axis='x', color='C0')
	ax.tick_params(axis='y', color='C0')

	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[np.max([0, t - 100]):(t + 1)])

	ax2.scatter(x, running_avg, color='C1')
	ax2.axes.get_xaxis().set_visible(False)
	ax2.yaxis.tick_right()
	ax2.set_ylabel('Score', color='C1')
	ax2.yaxis.set_label_position('right')
	ax2.tick_params(axis='y', color='C1')

	plt.savefig(filename)


class RepeatActionAndMaxFrame(gym.Wrapper):
	def __init__(self, env=None, repeat=4):
		super(RepeatActionAndMaxFrame, self).__init__(env)
		self.repeat = repeat
		self.shape = env.observation_space.low.shape
		self.frame_buffer = np.zeros_like((2, self.shape))

	def step(self, action):
		t_reward = 0.0
		done = False
		for i in range(self.repeat):
			obs, reward, done, info = self.env.step(action)
			t_reward += reward
			idx = i % 2
			self.frame_buffer[idx] = obs
			if done:
				break

		max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
		return max_frame, t_reward, done, info

	def reset(self, **kwargs):
		obs = self.env.reset()
		self.frame_buffer = np.zeros_like((2, self.shape))
		self.frame_buffer[0] = obs

		return obs


class PreprocessFrame(gym.ObservationWrapper):
	def __init__(self, shape, env=None):
		super(PreprocessFrame, self).__init__(env)
		self.shape = (shape[2], shape[0], shape[1])
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
												shape=self.shape, dtype=np.float32)

	def observation(self, obs):
		new_frame = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
		resized_screen = cv2.resize(new_frame, self.shape[1:],
									interpolation=cv2.INTER_AREA)
		new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
		new_obs = new_obs / 255.0

		return  new_obs