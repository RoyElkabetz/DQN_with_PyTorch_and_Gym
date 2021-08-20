import gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agents import Agent

env = gym.make('FrozenLake-v0')
n_games = 5000
win_prc_list = []
scores = []
agent = Agent(lr=0.001, gamma=0.9, eps_start=1, eps_end=0.01, eps_dec=0.9999995, n_states=16, n_actions=4)

for i in range(n_games):
	done = False
	obs = env.reset()
	score = 0

	while not done:
		action = agent.choose_action(obs)
		obs_, reward, done, info = env.step(action)
		agent.learn(obs, action, reward, obs_)
		score += reward
		obs = obs_
	scores.append(score)

	if i % 100 == 0:
		win_prc = np.mean(scores[-100:])
		win_prc_list.append(win_prc)
	if i % 1000 == 0:
		print('| game: {:10d}| win_prc: {:2.2f} | epsilon: {:2.8f} |'.format(i, win_prc, agent.epsilon))
env.close()

plt.figure()
plt.plot(win_prc_list)
plt.show()
