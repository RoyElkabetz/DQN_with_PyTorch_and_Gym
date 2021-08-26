# DQN variations using PyTorch and Jym

This repo contains a PyTorch written DQN, DDQN, DuelingDQN and DuelingDDQN implementations, following the next Google DeepMind's papers:

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (2015)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (2016)

## Background
In the first paper above (Human-level control through deep reinforcement learning (2005)) the authors state *"We set out to create a single algorithm that would be able to develop a wide range of competencies on a varied range of challenging tasks — a central goal of general artificial intelligence"*. Indeed, the main advantages in estimating the **Q-value function** using a Deep Neural Network (DNN) are, (1) An identical network can be used in a variety of very different games and sequential tasks, (2) The complexity of the training does not scale trivially with the size of the (state, action) space, which means that a very large (state, action) space can be modeled without a problem using a pretty small DNN (comparing to real-life applications solved using DNNs). In this repository, I followed the development of the DQN to DDQN and then to Dueling-DQN and Dueling-DDQN algorithms, and implemented all four of them as described in the papers. My goal was less to make a clean and clear API for DQN algorithms rather than to gain some fundamental understanding of the basic concepts that drove the DRL field forward in the last few years.

## Requirements and ROM installation

|Library         | Version |
|----------------|---------|
|`Python`        |  `3.8`  |
|`torch`         |  `1.8.1`|
|`gym`           | `0.18.3`|
|`numpy`         | `1.19.5`|

### ROMs installation
After installing the gym library, in order to render the games from the Atari library you need to install the Atari ROMs following the next few steps:
1. Download and save the Atari ROM files from the next [url](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).
2. Extract from the downloaded `Roms.rar` file the two zip files `HC ROMS.zip` and `ROMS.zip`.
3. Open a Terminal window.
4. Run the next command in the terminal `python -m atari_py.import_roms path_to_roms_folder\ROMS`

note: if your default python version is different from the one you will be using in working with gym (i.e python 2 as default bat you are working with python 3 , use `python3` instead of `python` in step (4)).

## Files Description

## API

## Train 

## Play

## Reference


 
