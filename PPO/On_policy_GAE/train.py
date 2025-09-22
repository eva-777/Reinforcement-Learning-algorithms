import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import gym

from ppo import Agent
from utils import log_message

############################### Initialize environment ###############################
# -------------------------------------------------------- #

# is_continuous = False
# env_name = "CartPole-v1"

# is_continuous = False
# env_name = "LunarLander-v2"

# is_continuous = True
# env_name = "Pendulum-v1"

# is_continuous = True
# env_name = "BipedalWalker-v3"

# -------------------------------------------------------- #

env = gym.make(env_name)
max_episode_steps = env._max_episode_steps
if is_continuous:
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
else:
    action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

random_seed = 7  # set random seed if required (0 -> no random)

############################### PPO hyperparameters ###############################
EPI_LEN = 400
MAX_TIMESTEPS = int(60e4)

K_EPOCHS = 40
BATCH_SIZE = int(EPI_LEN * 1)
UPDATE_INTERVAL = EPI_LEN * 4
PRINT_FREQ = UPDATE_INTERVAL * 2
LOG_FREQ = EPI_LEN * 2

GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2

LR_ACTOR = 3e-4
LR_CRITIC = 1e-3

use_orthogonal = True

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
log_message(log_path, 'Ep, Timestep, Update, Avg_return, Actor_loss, Entropy, Critic_loss')

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
print(f"total_timestep: {MAX_TIMESTEPS},  episode_length: {EPI_LEN}")
print(f"k_epochs: {K_EPOCHS},  batch_size: {BATCH_SIZE},  update_interval: {UPDATE_INTERVAL}")
print(f"actor_lr: {LR_ACTOR},  critic_lr: {LR_CRITIC}")
print(f"gamma: {GAMMA},  lambda: {LAMBDA},  eps_clip: {EPS_CLIP}")
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
agent = Agent(use_orthogonal, is_continuous, state_dim, action_dim, GAMMA, LAMBDA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCHS, BATCH_SIZE)

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
while timestep <= MAX_TIMESTEPS:
    state, _ = env.reset()
    ep_return = 0

    # One episode
    for t in range(EPI_LEN): 
        timestep += 1

        # Interact with env
        action, logprob, value = agent.take_action(state)
        next_state, reward, terminated, _, _ = env.step(action)
        truncated = True if timestep % EPI_LEN == 0 else False

        # To memory
        agent.replay_buffer.add_memory(state, action, logprob, reward, value, terminated, truncated)
        state = next_state
        ep_return += reward

        # Update
        if timestep % UPDATE_INTERVAL == 0:
            last_state = next_state
            actor_loss, actor_entropy, critic_loss = agent.update_policy(last_state)
            update_iter += 1

            actor_loss_history.append(actor_loss)
            actor_entropy_history.append(actor_entropy)
            critic_loss_history.append(critic_loss)

            avg_actor_loss = np.mean(actor_loss_history[-10:])
            avg_actor_entropy = np.mean(actor_entropy_history[-10:])
            avg_critic_loss = np.mean(critic_loss_history[-10:])

        # Log
        if timestep % LOG_FREQ == 0:
            log_avg_return = np.mean(episode_return_history[-10:])
            log_message(log_path, f"{i_episode}: "  # episode
                        f"{timestep}, "   # Total time step
                        f"{update_iter}, "  # Update Iteration
                        f"{log_avg_return:.3f} "  # Average Episode Return
                        )
        # Print
        if timestep % PRINT_FREQ == 0:
            print_avg_return = np.mean(episode_return_history[-10:])
            print(f"Ep : {i_episode} \t\t  Timestep : {timestep} \t\t  Avg_return : {print_avg_return: .1f}")
        
        # MDP terminates
        if terminated == True:
            break
    
    # Apped
    i_episode += 1
    episode_return_history.append(ep_return)
    timestep_history.append(timestep)

    # Save
    if timestep >= int(10e4) and log_avg_return > best_return+1:     
        best_return = log_avg_return
        agent.save_model(chkpt_path)

env.close()

end_time = datetime.now().replace(microsecond=0)
training_time = end_time - start_time
log_message(log_path, f"\nTraining time = {training_time}")

print("============================================================================================")
print(f"Total training time: {end_time - start_time}")
print(f"Log file saved at: {log_path}")
print(f"Model saved at: {chkpt_path}")
print("============================================================================================")


############################### plot_learning_curve ###############################
x = np.arange(len(episode_return_history))
running_avg = np.zeros(len(episode_return_history))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(episode_return_history[max(0, i-50):(i+1)])

plt.figure(figsize=(9, 5))
plt.title(f'{env_name} with {run_num}th run')
plt.xlabel('Timestep')
plt.ylabel('Return')
plt.plot(timestep_history, episode_return_history, label='Episode Return', c='pink')
plt.plot(timestep_history, running_avg, label='Average Return', c='red')
plt.grid(linewidth = 0.3)
plt.legend()
plt.savefig(plt_path)
