import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import gymnasium as gym

from PPO_continuous.ppo import PPO_continuous
from utils import log_message, exponential_moving_average

def evaluate_policy(env, agent, runs):
    total_reward = 0
    for i in range(runs):
        state, _ = env.reset()
        done = False
        while not done:
            # Take deterministic actions when evaluate
            action, _, _ = agent.take_action(state, deterministic=True)
            state_next, reward, terminate, truncated, _ = env.step(action)
            done = (terminate or truncated)

            state = state_next
            total_reward += reward   

    return total_reward/runs


def main(args, env_name):
    ##################################### make Env #####################################
    # seed everything
    if args.random_seed:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False 
    
    # make Env
    env_train = gym.make(env_name)
    env_eval = gym.make(env_name)
    _, _ = env_train.reset(seed=args.random_seed)  # set seed for env_train
    _, _ = env_eval.reset(seed=int(args.random_seed+100))  # set seed for env_eval. NOTE: In fact, the seeds should be fixed and diverse, before training
    max_episode_steps = env_train._max_episode_steps

    # Continuous or not, state_dim, action_dim
    args.is_continuous = True
    args.state_dim = env_train.observation_space.shape[0]
    args.action_dim = env_train.action_space.shape[0]
    args.action_max = env_train.action_space.high[0]


    ################################# log, save, plot #################################
    current_path = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_path, f'result_{env_name}')
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)

    # -------------- log path, log information -------------- #
    log_dir = result_dir + '/log'
    if not os.path.exists(log_dir): 
        os.makedirs(log_dir)
    current_log_files = next(os.walk(log_dir))[2]
    run_num = int(len(current_log_files)/2)  
    train_log_path = os.path.join(log_dir, f'train_log_{run_num}.csv')
    eval_log_path = os.path.join(log_dir, f'eval_log_{run_num}.csv')

    # log initial training information
    init_info = (
        f"env={env_name}, training seed: {args.random_seed}, is_continuous: {args.is_continuous} \n"
        f"max_steps: {args.max_timesteps}, max_episode_steps: {max_episode_steps} \n"
        f"k_epoch: {args.K_epochs}, batch_size: {args.batch_size}, mini_batch: {args.mini_batch}, \n"
        f"gamma: {args.gamma}, lamda: {args.lamda}, eps_clip: {args.eps_clip} \n"
        f"lr_actor: {args.lr_actor}, lr_critic: {args.lr_critic}, lr_decay: {args.use_lr_decay} \n"
        f"orthogonal: {args.use_orthogonal}, entropy_coef: {args.entropy_coef}, entropy_decay: {args.entropy_decay} \n"
        f"adv_norm: {args.use_adv_norm}, grad_clip: {args.use_grad_clip}, value_clip: {args.use_value_clip} \n")
    log_message(train_log_path, "Episode, Timestep, Update, Avg_reward", timestamp=True)
    log_message(eval_log_path, init_info, timestamp=True)
    log_message(eval_log_path, "Episode, Timestep, Update, Avg_reward")
    
    # -------------- save path of checkpoints -------------- #
    chkpt_dir = result_dir + '/save'
    if not os.path.exists(chkpt_dir): 
        os.makedirs(chkpt_dir)
    chkpt_path = os.path.join(chkpt_dir, f'actor_{run_num}.pth')

    # -------------- plot path of trainning curve -------------- #
    plot_dir = result_dir + '/plot'
    if not os.path.exists(plot_dir): 
        os.makedirs(plot_dir)  
    plt_path = os.path.join(plot_dir, f'Avg_ep_reward_{run_num}.png')


    ############################### Training procedure ###############################
    agent = PPO_continuous(args)

    # record list
    train_reward_list = []      # for each episode
    train_timestep_list = []    # for each episode
    eval_reward_list = []       # for log eval
    best_avg_reward = -float('inf')
    log_avg_reward = -float('inf') 

    actor_loss_history = []
    actor_entropy_history = []
    critic_loss_history = []

    num_episode = 0
    num_timestep = 0
    num_update_iter = 0

    # training loop
    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("--------------------------------------------------------------------------------------------")

    while num_timestep <= args.max_timesteps:
        num_episode += 1
        state, _ = env_train.reset()
        episode_reward = 0
        done = False

        # one episode
        while not done:
            num_timestep += 1

            # interact with env
            action, logprob, value = agent.take_action(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env_train.step(action)
            done = (terminated or truncated)  # dead&win or truncation

            # to memory
            agent.buffer.add_memory(state, action, logprob, reward, value, terminated, done)
            state = next_state
            episode_reward += reward

            # update
            if num_timestep % args.batch_size == 0:
                num_update_iter += 1
                last_state = next_state
                actor_loss, actor_entropy, critic_loss = agent.update_policy(last_state, num_timestep)

                # actor_loss_list.append(actor_loss)
                # actor_entropy_list.append(actor_entropy)
                # critic_loss_list.append(critic_loss)

                # avg_actor_loss = np.mean(actor_loss_list[-10:])
                # avg_actor_entropy = np.mean(actor_entropy_list[-10:])
                # avg_critic_loss = np.mean(critic_loss_list[-10:])
                # print(avg_actor_loss, avg_actor_entropy, avg_critic_loss)

            # log training
            if num_timestep % args.log_freq == 0:
                log_avg_reward = np.mean(train_reward_list[-10:])
                log_message(train_log_path, f"{num_episode}: "      # current episode
                                    f"{num_timestep}, "             # current timestep
                                    f"{num_update_iter}, "          # current update iteration
                                    f"{log_avg_reward :.3f} "       # average episode reward during training
                                    )
            
            # evaluate, log evaluation
            if num_timestep % args.eval_freq == 0:
                eval_avg_reward = evaluate_policy(env_eval, agent, runs=5)
                eval_reward_list.append(eval_avg_reward)
                log_message(eval_log_path, f"{num_episode}: "
                                    f"{num_timestep}, "           
                                    f"{num_update_iter}, "        
                                    f"{eval_avg_reward :.3f} "    # average episode reward during evaluation
                                    )
                print(f"Ep : {num_episode} \t\t  Timestep : {num_timestep} \t\t  Eval_avg_reward : {eval_avg_reward:.2f}")

                # save
                # NOTE: best_avg_reward may not mean the best model during training, unless the eval seeds are fixed.
                if num_timestep >= int(0.2*args.max_timesteps) and eval_avg_reward >= best_avg_reward:     
                    best_avg_reward = eval_avg_reward
                    agent.save_model(chkpt_path)
                    print("--------------------------------------------------------------")
                    print(f" model saved, with best average reward = {best_avg_reward :.2f} ")
                    print("--------------------------------------------------------------")
            
        # apped
        train_reward_list.append(episode_reward)
        train_timestep_list.append(num_timestep)

    env_train.close()
    env_eval.close()

    end_time = datetime.now().replace(microsecond=0)
    training_time = end_time - start_time
    log_message(eval_log_path, f"\nBest Avg_reward = {best_avg_reward :.3f}")
    log_message(eval_log_path, f"\nTraining time = {training_time}")
    print("--------------------------------------------------------------------------------------------")
    print(f"Ended training at: {end_time}")
    print(f"Total training time: {end_time - start_time}")
    print(f"Log file saved at: {eval_log_path}")
    print(f"Model saved at: {chkpt_path}")
    print("============================================================================================")


    ############################### plot_learning_curve ###############################
    eval_timestep_array = np.arange(args.eval_freq, args.max_timesteps, args.eval_freq)
    eval_reward_array = np.array(eval_reward_list)
    alpha = 0.05
    smoothed_train_reward = exponential_moving_average(train_reward_list, alpha)

    plt.figure(figsize=(10, 6))
    plt.plot(train_timestep_list, train_reward_list, color='lightblue', alpha=0.7, linewidth=1, label='Train Episode reward')
    plt.plot(train_timestep_list, smoothed_train_reward, color='blue', linewidth=1, label='EMA (alpha=0.05)')
    plt.plot(eval_timestep_array, eval_reward_array, color='black', linewidth=1, label='Eval episode reward')
    plt.title(f'Log of {env_name} with run_num={run_num}', fontsize=13)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Average episode reward', fontsize=12)
    plt.xlim(0, args.max_timesteps)
    plt.ylim(min(train_reward_list)-50, max(train_reward_list)+50)  # 留出边距
    plt.legend(loc='lower right')
    plt.grid(True, color='gray', linestyle='--', alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.savefig(plt_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")
    # Training hyperparameters
    parser.add_argument('--max_timesteps', type=int, default=int(200e4), help='Max steps for training')
    parser.add_argument('--batch_size', type=int, default=2048, help='Update interval')
    parser.add_argument('--mini_batch', type=int, default=64, help='Length of trajectory in mini-batch')
    parser.add_argument('--K_epochs', type=int, default=10, help='Update agent for K epochs')
    parser.add_argument('--log_freq', type=int, default=2048*2, help='Training log interval')
    parser.add_argument('--eval_freq', type=int, default=2048*4, help='Model evaluation interval')
    parser.add_argument('--save_freq', type=int, default=2048*50, help='Model save interval')
    parser.add_argument('--random_seed', type=int, default=7, help='Random seed for env_train, 0 -> no random')
    # Agent hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden net width of actor and critic')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor, [0.95, 0.999]')
    parser.add_argument('--lamda', type=float, default=0.95, help='GAE Factor, [0.95, 0.99]')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip rate for old_actor and new_actor')
    parser.add_argument('--lr_actor_decay', type=float, default=1e-4, help='Decay limit of lr_actor')
    parser.add_argument('--lr_critic_decay', type=float, default=2e-4, help='Decay limit of lr_critic')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Learning rate of actor, [1e-4, 3e-4]')
    parser.add_argument('--lr_critic', type=float, default=6e-4, help='Learning rate of critic, [1e-4, 4e-3]')
    # Trick switch
    parser.add_argument('--use_orthogonal', type=bool, default=True, help='orthogonal initialization')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='coefficient of entropy loss')
    parser.add_argument('--entropy_decay', type=float, default=0.99, help='Coefficient of entropy loss')
    parser.add_argument('--use_adv_norm', type=bool, default=True, help='advantage normalization')
    parser.add_argument('--use_grad_clip', type=bool, default=True, help=' gradient clip')
    parser.add_argument('--use_value_clip', type=bool, default=False, help=' value clip for critic')
    parser.add_argument('--use_lr_decay', type=bool, default=False, help=' learning rate decay')
    
    args = parser.parse_args()
    print(args)

    # ------------------------ Train ------------------------ #

    # env_name = "Pendulum-v1"  # max_episode_steps = 200
    env_name = "BipedalWalker-v3"  # max_episode_steps = 1600

    main(args, env_name)