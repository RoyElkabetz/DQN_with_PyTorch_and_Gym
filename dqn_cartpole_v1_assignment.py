import gym
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agents import DQNAgent
from utils import plot_learning_curve


env = gym.make('CartPole-v1')
n_games = 5000
win_prc = []
scores = []
eps_history = []


agent = DQNAgent(input_dims=env.observation_space.shape, n_actions=env.action_space.n)

for i in range(n_games):
	done = False
	obs = env.reset()
	score = 0

	while not done:
		action = agent.choose_action(obs)
		obs_, reward, done, info = env.step(action)
		score += reward
		agent.learn(obs, action, reward, obs_)
		obs = obs_
	scores.append(score)
	eps_history.append(agent.epsilon)
	if i % 100 == 0:
		avg_score = np.mean(scores[-100:])
		print('| game: {:10d}| score: {:2.2f} | avg_score: {:2.2f} | epsilon: {:2.8f} |'.format(i, score, avg_score, agent.epsilon))
env.close()

filename = 'cartpole_naive_dqn.png'
x = [i + 1 for i in range(n_games)]
plot_learning_curve(x, scores, eps_history, filename)
		




