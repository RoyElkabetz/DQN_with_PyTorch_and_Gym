import argparse, os
import gym
from gym import wrappers
import numpy as np
from utils import plot_learning_curve, make_env
import agents as Agents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q learning algorithms implementation')

    # Arguments
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for the update rule')
    parser.add_argument('-epsilon', type=float, default=1.0,
                        help='Initial epsilon value for the epsilon-greedy policy')
    parser.add_argument('-lr', type=float, default=0.0004,
                        help='The learning rate')
    parser.add_argument('-mem_size', type=int, default=20000,
                        help='The maximal memory size used for storing transitions (replay buffer)')  # ~ 6 GB RAM
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size for learning')
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Lower limit for epsilon')
    parser.add_argument('-eps_dec', type=float, default=5e-7,
                        help='Value for epsilon linear decrement')
    parser.add_argument('-replace', type=int, default=1000,
                        help='Number of learning steps for target network replacement')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                        help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    parser.add_argument('-env_name', type=str, default='PongNoFrameskip-v4',
                        help='Atari environments.\nPongNoFrameskip-v4 \
                                                 \nBreakoutNoFrameskip-v4 \
                                                 \nSpaceInvadersNoFrameskip-v4 \
                                                 \nEnduroNoFrameskip-v4 \
                                                 \nAtlantisNoFrameskip-v4')
    parser.add_argument('-path', type=str, default='tmp/',
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

    if args.load_checkpoint:
        agent.load_models()

    if args.monitor:
        env = wrappers.Monitor(env, 'videos/', video_callable=lambda episode_id: True, force=True)

    fname = agent.algo + '_' + agent.env_name + '_lr_' + str(agent.lr) + '_' + str(args.n_games) + '_games'
    figure_file = 'plots/' + fname + '.png'
    scores_file = fname + '_scores.npy'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if args.render:
                env.render()
            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()

            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %.1f best score %.1f epsilon %.2f' %
              (avg_score, best_score, agent.epsilon), 'steps ', n_steps)

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
    np.save(scores_file, np.array(scores))
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
