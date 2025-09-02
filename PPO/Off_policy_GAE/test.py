import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool
from PPO import Agent


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    env_name = "Pendulum-v1"
    is_continuous = True

    TOTAL_EP = 10     # total num of testing episodes
    EPI_LEN = 300            # max timesteps in one episode

    render = True               # render environment on screen
    frame_delay = 0.01             # if required; add delay b/w frames

    K_EPOCHS = 60
    BATCH_SIZE = 300

    GAMMA = 0.99
    LAMBDA = 0.95
    EPS_CLIP = 0.2

    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3

    #####################################################

    env = gym.make(env_name, render_mode='human')

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if is_continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    agent = Agent(is_continuous, state_dim, action_dim, GAMMA, LAMBDA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCHS, BATCH_SIZE)

    # preTrained weights file path
    run_num = 1
    checkpoint_path = f"./PPO_v2/ppo_save/Pendulum-v1/actor_Pendulum-v1_{run_num}.pth"
    print("loading network from: " + checkpoint_path)

    agent.load_model(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, TOTAL_EP+1):
        ep_reward = 0
        state, _ = env.reset()

        for t in range(1, EPI_LEN+1):
            action, _, _ = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / TOTAL_EP
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
