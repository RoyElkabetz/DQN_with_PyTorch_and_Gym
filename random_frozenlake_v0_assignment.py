import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
n_games = 1000
scc_prc = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    scores.append(score)    
    
    if i % 10 == 0:
    	scc_prc.append(np.mean(scores[-10:]) * 100)
env.close()

plt.figure()
plt.plot(scc_prc)
plt.show()

# env = gym.make('FrozenLake-v0')
# print(env.action_space)
# print(env.observation_space)

