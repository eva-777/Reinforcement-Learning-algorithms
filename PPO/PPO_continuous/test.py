import os
import time
import argparse
from datetime import datetime
import gymnasium as gym

from PPO_continuous.ppo import PPO_continuous


#################################### Test ###################################
def test(args, env_name, run_num):

    ##################################### Make Env #####################################
    env = gym.make(env_name, render_mode = 'human' if args.render else None)
    _, _ = env.reset(seed=args.random_seed if args.random_seed else None)
    max_episode_steps = env._max_episode_steps

    # state_dim, action_dim
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_max = env.action_space.high[0]

    # load agent
    current_fold = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_fold, f'result_{env_name}/save/actor_{run_num}.pth')
    agent = PPO_continuous(args)
    agent.load_model(checkpoint_path)
    print(f"loading network from: {checkpoint_path}")

    print("--------------------------------------------------------------------------------------------")
    total_reward = 0
    for ep in range(args.total_episodes):
        episode_reward = 0
        state, _ = env.reset()

        for t in range(max_episode_steps):
            action, _, _ = agent.take_action(state, deterministic=True)
            state, reward, terminate, truncated, _ = env.step(action)
            done = (terminate or truncated)
            episode_reward += reward

            time.sleep(args.frame_delay)

            if done:
                break

        total_reward += episode_reward
        print(f'Episode: {ep+1} \t\t Return: {episode_reward :.2f}')
        episode_reward = 0

    env.close()

    print("--------------------------------------------------------------------------------------------")
    avg_test_reward = total_reward / args.total_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print(f"average test reward : {avg_test_reward}")
    print("============================================================================================")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")   
    # Test hyperparameters
    parser.add_argument('--total_episodes', type=int, default=40, help='Total num of testing episodes')
    parser.add_argument('--random_seed', type=int, default=77, help='Random seed for env, 0 -> no random')
    parser.add_argument('--render', type=bool, default=True, help='Render or Not')
    parser.add_argument('--frame_delay', type=float, default=0, help='frame delay on screen when render')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden net width of actor and critic')
    args = parser.parse_args()
    print(args)

    # ------------------------ Test ------------------------ #

    # env_name = "Pendulum-v1"
    env_name = "BipedalWalker-v3"
    run_num = 4

    test(args, env_name, run_num)
