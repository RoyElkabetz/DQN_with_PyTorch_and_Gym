import gym 
import numpy as np
import matplotlib.pyplot as plt

def det_policy(observation):
	# [0, 1, 2, 3] = [<, v, >, ^]
	if observation in [0, 2, 4, 6, 10]:
		return 1
	else:
		return 2

env = gym.make('FrozenLake-v0')
n_games = 10000
win_prc = []
scores = []

for i in range(n_games):
	done = False
	score = 0
	obs = env.reset()
	while not done:
		# env.render()
		action = det_policy(obs)
		# action = env.action_space.sample()
		obs, reward, done, info = env.step(action)
		score += reward
	scores.append(score)
	if i % 10 == 0:
		win_prc.append(np.mean(scores[-10:]))
env.close()

plt.figure()
plt.plot(win_prc)
plt.show()

