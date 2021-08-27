import argparse, os
import gym
from gym import wrappers
import numpy as np
from utils import plot_learning_curve, make_env
import agents as Agents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q learning algorithms implementation')

    # Arguments
    parser.add_argument('-train', type=bool, default=False,
                        choices=[True, False],
                        help='Choosing the mode of the agent, True for training or False for playing.')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for the update rule')
    parser.add_argument('-epsilon', type=float, default=1.0,
                        help='Initial epsilon value for the epsilon-greedy policy')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='The learning rate')
    parser.add_argument('-mem_size', type=int, default=20000,
                        help='The maximal memory size used for storing transitions (replay buffer)')  # ~ 6 GB RAM
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size for learning')
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Lower limit for epsilon')
    parser.add_argument('-eps_dec', type=float, default=1e-5,
                        help='Value for epsilon linear decrement')
    parser.add_argument('-replace', type=int, default=1000,
                        help='Number of learning steps for target network replacement')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                        choices=['DQNAgent',
                                 'DDQNAgent',
                                 'DuelingDQNAgent',
                                 'DuelingDDQNAgent'],
                        help='choose from the next DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    parser.add_argument('-env_name', type=str, default='PongNoFrameskip-v4',
                        choices=['PongNoFrameskip-v4',
                                 'BreakoutNoFrameskip-v4',
                                 'SpaceInvadersNoFrameskip-v4',
                                 'EnduroNoFrameskip-v4',
                                 'AtlantisNoFrameskip-v4'],
                        help='choose from the next Atari environments:\
                                                 \nPongNoFrameskip-v4 \
                                                 \nBreakoutNoFrameskip-v4 \
                                                 \nSpaceInvadersNoFrameskip-v4 \
                                                 \nEnduroNoFrameskip-v4 \
                                                 \nAtlantisNoFrameskip-v4')
    parser.add_argument('-path', type=str, default='models/',
                        help='Path for loading and saving models')
    parser.add_argument('-n_games', type=int, default=1,
                        help='Number of games for the Agent to play')
    parser.add_argument('-skip', type=int, default=4,
                        help='Number of environment frames to stack')
    parser.add_argument('-gpu', type=str, default='0',
                        help='CPU: 0, GPU: 1')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='Load a model checkpoint')
    parser.add_argument('-render', type=bool, default=False,
                        help='Render the game to screen ? True/False')
    parser.add_argument('-monitor', type=bool, default=False,
                        help='If True, a video is being saved for each episode')

    args = parser.parse_args()

    # arrange work between two GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    env = make_env(env_name=args.env_name)
    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma,
                   epsilon=args.epsilon,
                   lr=args.lr,
                   n_actions=env.action_space.n,
                   input_dims=env.observation_space.shape,
                   mem_size=args.mem_size,
                   batch_size=args.bs,
                   eps_min=args.eps_min,
                   eps_dec=args.eps_dec,
                   replace=args.replace,
                   algo=args.algo,
                   env_name=args.env_name,
                   chkpt_dir=args.path)

    if args.monitor:
        env = wrappers.Monitor(env, 'videos/', video_callable=lambda episode_id: True, force=True)

    # create name strings for saving data
    fname = agent.algo + '_' + agent.env_name + '_lr_' + str(agent.lr) + '_' + str(args.n_games) + '_games'
    figure_file = 'plots/' + fname + '.png'
    scores_file = 'scores/' + fname + '_scores.npy'
    steps_file = 'scores/' + fname + '_steps.npy'
    eps_history_file = 'scores/' + fname + '_eps_history.npy'

    n_steps = 0
    games_played = 0
    scores, eps_history, steps_array = [], [], []

    if args.load_checkpoint:
        # load Q models
        agent.load_models()

        if args.train:
            # load old scores and related data
            with np.load(scores_file) as scores_data:
                scores = list(scores_data)
                games_played = len(scores)
                for t in range(len(scores)):
                    t_avg_score = np.mean(scores[np.max([0, t - 100]):(t + 1)])
                    if t_avg_score > best_score:
                        best_score = t_avg_score

            with np.load(steps_file) as steps_data:
                steps_array = list(steps_data)
                n_steps = steps_data[-1]

            with np.load(eps_history_file) as eps_data:
                eps_history = list(eps_data)
                agent.epsilon = eps_history[-1]

    # training / playing
    for i in range(games_played, args.n_games + games_played):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if args.render:
                env.render()
            if args.train:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()

            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %.1f best average score %.1f epsilon %.2f' %
              (avg_score, best_score, agent.epsilon), 'steps ', n_steps)

        if avg_score > best_score:
            if args.train:
                agent.save_models()
            best_score = avg_score

    # save training data
    if args.train:
        np.save(scores_file, np.array(scores))
        np.save(steps_file, np.array(steps_array))
        np.save(eps_history_file, np.array(eps_history))

        # plot the learning curve
        plot_learning_curve(steps_array, scores, eps_history, figure_file)
