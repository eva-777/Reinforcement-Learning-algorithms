import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

import torch
import gym

from ppo import Agent
from utils import log_message, exponential_moving_average

############################### Initialize environment ###############################
# -------------------------------------------------------- #

# is_continuous = False
# env_name = "CartPole-v1"  # max_episode_steps = 500

is_continuous = False
env_name = "LunarLander-v2"  # max_episode_steps = 1000

# is_continuous = True
# env_name = "Pendulum-v1"  # max_episode_steps = 200

# is_continuous = True
# env_name = "BipedalWalker-v3"  # max_episode_steps = 1600

# -------------------------------------------------------- #

env = gym.make(env_name)
max_episode_steps = env._max_episode_steps
if is_continuous:
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
else:
    action_dim = env.action_space.n
    action_high = None
state_dim = env.observation_space.shape[0]

random_seed = 7  # set random seed if required (0 -> no random)

############################### PPO hyperparameters ###############################
max_timesteps = int(100e4)
max_ep_len = max_episode_steps
k_epoch = 10
mini_batch = 64  # mini_batch_size
update_interval = 2048  # batch_size
log_freq = update_interval * 2
print_freq = update_interval * 4

gamma = 0.99
lamda = 0.95  # 0.95 ~ 0.99
eps_clip = 0.2  # 0.1 ~ 0.3
lr_actor = 3e-4  # 1e-4 ~ 3e-4
lr_critic = 1e-3  # 1e-4 ~ 4e-3

use_entropy = False
use_orthogonal = True
use_grad_clip = True
use_value_clip = False
use_lr_decay = False
use_adv_norm = False

################################# log, save, plot #################################
env_dir = f'./On_policy_GAE/result_{env_name}'
if not os.path.exists(env_dir):
          os.makedirs(env_dir)
run_num = 0

### log ###
log_dir = env_dir + '/log'
if not os.path.exists(log_dir):
          os.makedirs(log_dir)

current_files = next(os.walk(log_dir))[2]
run_num = len(current_files)
log_path = os.path.join(log_dir, f'training_log_{run_num}.csv')

# log initial training information
log_message(log_path, f"Env={env_name}", timestamp=True)
log_message(log_path, 'Ep, Timestep, Update, Avg_return')

### save ###
chkpt_dir = env_dir + '/save'
if not os.path.exists(chkpt_dir):
          os.makedirs(chkpt_dir)
chkpt_path = os.path.join(chkpt_dir, f'actor_{run_num}.pth')

### plot ###
plot_dir = env_dir + '/plot'
if not os.path.exists(plot_dir):
          os.makedirs(plot_dir)

plt_path = os.path.join(plot_dir, f'Episode_return_{env_name}_{run_num}.png')

############################ Print training information ############################
print("============================================================================================")
print(f"environment: {env_name},  is_continuous: {is_continuous}")
print(f"total_timestep: {max_timesteps},  episode_length: {max_ep_len}")
print(f"k_epochs: {k_epoch},  mini_batch: {mini_batch},  update_interval: {update_interval}")
print(f"actor_lr: {lr_actor},  critic_lr: {lr_critic}")
print(f"gamma: {gamma},  lambda: {lamda},  eps_clip: {eps_clip}")
print(f"use_orthogonal_init: {use_orthogonal}, use_value_clip: {use_value_clip}, use_grad_clip: {use_grad_clip}")
print(f"use_entropy: {use_entropy}, use_lr_decay: {use_lr_decay}, use_adv_norm: {use_adv_norm}, ...")
if random_seed:
    print(f"random seed: {random_seed}")
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
print("============================================================================================")

############################### Training procedure ###############################
agent = Agent(use_orthogonal, use_grad_clip, use_value_clip, is_continuous, state_dim, action_dim, gamma, lamda, eps_clip, lr_actor, lr_critic, k_epoch, mini_batch,max_timesteps, action_high)

episode_return_history = []
timestep_history = []
best_return = env.reward_range[0] + 1

actor_loss_history = []
actor_entropy_history = []
critic_loss_history = []

i_episode = 0
timestep = 0
update_iter = 0
print_avg_return = 0
log_avg_return = 0

# Training loop
start_time = datetime.now().replace(microsecond=0)
while timestep <= max_timesteps:
    state, _ = env.reset()
    ep_return = 0
    done = False

    # One episode
    while not done:
        timestep += 1

        # Interact with env
        action, logprob, value = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = (terminated or truncated)  # dead&win or truncation

        # To memory
        agent.replay_buffer.add_memory(state, action, logprob, reward, value, terminated, truncated)
        state = next_state
        ep_return += reward

        # Update
        if timestep % update_interval == 0:
            last_state = next_state  # 
            actor_loss, actor_entropy, critic_loss = agent.update_policy(last_state, timestep)
            update_iter += 1

            actor_loss_history.append(actor_loss)
            actor_entropy_history.append(actor_entropy)
            critic_loss_history.append(critic_loss)

            avg_actor_loss = np.mean(actor_loss_history[-10:])
            avg_actor_entropy = np.mean(actor_entropy_history[-10:])
            avg_critic_loss = np.mean(critic_loss_history[-10:])

        # Log
        if timestep % log_freq == 0:
            log_avg_return = np.mean(episode_return_history[-10:])
            log_message(log_path, f"{i_episode}: "  # Current Episode
                                f"{timestep}, "   #  Current Timestep
                                f"{update_iter}, "  # Update Iteration
                                f"{log_avg_return:.3f} "  # Average Episode Return
                                )
        # Print
        if timestep % print_freq == 0:
            print_avg_return = np.mean(episode_return_history[-10:])
            print(f"Ep : {i_episode} \t\t  Timestep : {timestep} \t\t  Avg_return : {print_avg_return: .1f}")
        
    
    # Apped
    i_episode += 1
    episode_return_history.append(ep_return)
    timestep_history.append(timestep)

    # Save
    if timestep >= int(0.6*max_timesteps) and log_avg_return > best_return+1:     
        best_return = log_avg_return
        agent.save_model(chkpt_path)

env.close()

end_time = datetime.now().replace(microsecond=0)
training_time = end_time - start_time
log_message(log_path, f"\nTraining time = {training_time}")

print("============================================================================================")
print(f"Started training at: {start_time}")
print(f"Total training time: {end_time - start_time}")
print(f"Log file saved at: {log_path}")
print(f"Model saved at: {chkpt_path}")
print("============================================================================================")


############################### plot_learning_curve ###############################
timesteps = np.array(timestep_history)
returns = np.array(episode_return_history)
alpha = 0.05
smoothed_returns = exponential_moving_average(returns, alpha)

plt.figure(figsize=(8, 5))
plt.plot(timesteps, returns, color='pink', alpha=0.7, linewidth=1, label='Episode return')
plt.plot(timesteps, smoothed_returns, color='red', linewidth=1, label=f'EMA (alpha={alpha})')
plt.title(f'Return - {env_name}', fontsize=14)
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Return', fontsize=12)
plt.xlim(0, max(timesteps))
plt.ylim(min(returns)-50, max(returns)+50)  # 留出边距
plt.legend(loc='lower right')
plt.grid(True, color='gray', linestyle='--', alpha=0.3)
plt.gca().set_axisbelow(True)
plt.savefig(plt_path)
plt.show()
