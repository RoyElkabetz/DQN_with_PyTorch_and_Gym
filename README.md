# DQN variations using PyTorch and Jym

This repo contains a PyTorch written DQN, DDQN, DuelingDQN and DuelingDDQN implementations, following the next Google DeepMind's papers:

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (2015)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (2016)

## Background
In the first paper above (Human-level control through deep reinforcement learning (2005)) the authors state *"We set out to create a single algorithm that would be able to develop a wide range of competencies on a varied range of challenging tasks — a central goal of general artificial intelligence"*. Indeed, the main advantages in estimating the **Q-value function** using a Deep Neural Network (DNN) are, (1) An identical network can be used in a variety of very different games and sequential tasks, (2) The complexity of the training does not scale trivially with the size of the (state, action) space, which means that a very large (state, action) space can be modeled without a problem using a pretty small DNN (comparing to real-life applications solved using DNNs). In this repository, I followed the development of the DQN to DDQN and then to Dueling-DQN and Dueling-DDQN algorithms, and implemented all four of them as described in the papers. My goal was less to make a clean and clear API for DQN algorithms rather than to gain some fundamental understanding of the basic concepts that drove the DRL field forward in the last few years.

## Requirements and ROM installation

## Files Description

## API

## Train 

## Play

## Reference


 
