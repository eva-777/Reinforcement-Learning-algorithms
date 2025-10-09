import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool
from ppo import Agent


#################################### Testing ###################################
def test(env_name, run_num):
    print("============================================================================================")

    if env_name in ["CartPole-v1", "LunarLander-v2"]:
        is_continuous = False
    elif env_name in ["Pendulum-v1", "BipedalWalker-v3"]:
        is_continuous = True

    ################## hyperparameters ##################

    total_episode = 5     # total num of testing episodes

    render = True               # render environment on screen
    frame_delay = 0.001             # if required; add delay b/w frames

    max_timesteps = None
    k_epoch = 60
    batch_size = 300
    gamma = 0.99
    lamda = 0.95
    eps_clip = 0.2
    lr_actor = 3e-4
    lr_critic = 1e-3
    use_orthogonal = False
    use_value_clip = False

    #####################################################

    env = gym.make(env_name, render_mode='human')
    max_episode_steps = env._max_episode_steps
    state_dim = env.observation_space.shape[0]
    if is_continuous:
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high[0]
    else:
        action_dim = env.action_space.n
        action_high = None

    # Initialize agent
    agent = Agent(use_orthogonal, use_value_clip, is_continuous, state_dim, action_dim, gamma, lamda, eps_clip, lr_actor, lr_critic, k_epoch, batch_size, max_timesteps, action_high)

    # PreTrained weights file path
    checkpoint_path = f"./On_policy_GAE/result_{env_name}/save/actor_{run_num}.pth"
    print("loading network from: " + checkpoint_path)

    agent.load_model(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(total_episode):
        ep_return = 0
        state, _ = env.reset()

        for t in range(max_episode_steps):
            action, _, _ = agent.take_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_return += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        test_running_reward +=  ep_return
        print('Episode: {} \t\t Return: {}'.format(ep+1, round(ep_return, 2)))
        ep_return = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_episode
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':


    # env_name = "CartPole-v1"
    # env_name = "LunarLander-v2"
    # env_name = "Pendulum-v1"
    env_name = "BipedalWalker-v3"
    
    run_num = 7

    test(env_name, run_num)
