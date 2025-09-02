import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import gym

from PPO import Agent


####### initialize environment hyperparameters #######

# env_name = "CartPole-v1"  # truncation = 500
# is_continuous = False

env_name = "Pendulum-v1"  # truncation = 200
is_continuous = True

env = gym.make(env_name)

if is_continuous:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

################ PPO hyperparameters ################
NUM_EPISODE = 500
EPI_LEN = 600
MAX_TIMESTEPS = NUM_EPISODE*EPI_LEN

K_EPOCHS = 80
BATCH_SIZE = EPI_LEN
UPDATE_INTERVAL = BATCH_SIZE * 4

GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2

LR_ACTOR = 3e-4
LR_CRITIC = 1e-3


############### print, logging, saving ###############
### printing ###
print_freq = 4

### logging ###
log_dir = './On_policy_GAE/ppo_log'
if not os.path.exists(log_dir):
          os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name
if not os.path.exists(log_dir):
          os.makedirs(log_dir) 

run_num = 0
current_files = next(os.walk(log_dir))[2]
run_num = len(current_files)

log_path = os.path.join(log_dir, 'ppo_{}_{}.csv'.format(env_name, run_num))
log_file = open(log_path, "w+")
log_file.write('episode, timestep, update_iters, epi_return, avg_turn\n')

### save ###
chkpt_dir = './On_policy_GAE/ppo_save'
if not os.path.exists(chkpt_dir):
          os.makedirs(chkpt_dir)
chkpt_dir = chkpt_dir + '/' + env_name
if not os.path.exists(chkpt_dir):
          os.makedirs(chkpt_dir) 

chkpt_path = os.path.join(chkpt_dir, 'actor_{}_{}.pth'.format(env_name, run_num))


########### printing training information ###########
print("============================================================================================")
print(f"environment: {env_name},  is_continuous: {is_continuous}")
print(f"total_episodes: {NUM_EPISODE},  episode_length: {EPI_LEN}")
print(f"k_epochs: {K_EPOCHS},  batch_size: {BATCH_SIZE},  update_interval: {UPDATE_INTERVAL}")
print(f"actor_lr: {LR_ACTOR},  critic_lr: {LR_CRITIC}")
print(f"gamma: {GAMMA},  lambda: {LAMBDA},  eps_clip: {EPS_CLIP}")
print("============================================================================================")


################# training procedure ################
agent = Agent(is_continuous, state_dim, action_dim, GAMMA, LAMBDA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCHS, BATCH_SIZE)

best_return = env.reward_range[0] + 1
return_history = []

update_iter = 0
avg_return = 0
time_step = 0

start_time = datetime.now().replace(microsecond=0)

# training loop
for ep_i in range(NUM_EPISODE):
    state, _ = env.reset()
    ep_return = 0

    # one episode
    for t in range(EPI_LEN):
        
        # 收集数据
        action, logprob, value = agent.select_action(state)
        state_next, reward, done, truncation, _ = env.step(action)
        
        # 记录数据
        agent.replay_buffer.add_memory(state, action, logprob, reward, value, done)
        state = state_next
        ep_return += reward

        time_step += 1

        # 更新策略
        if time_step % UPDATE_INTERVAL == 0:
            last_state = state_next
            agent.update_policy(last_state)

            update_iter += 1

        # Episode结束
        if done == True:
            break

    # 记录日志
    return_history.append(ep_return)
    avg_return = np.mean(return_history[-100:])
    log_file.write('{}, {}, {}, {}, {}\n'.format(ep_i+1, time_step, update_iter, ep_return, avg_return))
    
    # 打印训练信息
    if (ep_i+1) % print_freq == 0:
        print(f"Ep {ep_i+1}, Timestep {time_step}, Update {update_iter}, Ep_return {ep_return: .1f}, Avg_return {avg_return: .1f}")

    # 保存模型
    if ep_i >= 100 and avg_return > best_return:     
        best_return = avg_return
        agent.save_model(chkpt_path)

 
log_file.close()
env.close()

end_time = datetime.now().replace(microsecond=0)
print("============================================================================================")
print(f"Total training time: {end_time - start_time}")
print(f"Log file saved at: {log_path}")
print(f"Model saved at: {chkpt_path}")
print("============================================================================================")


############### plot_learning_curve ###############
x = np.arange(len(return_history))
running_avg = np.zeros(len(return_history))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(return_history[max(0, i-50):(i+1)])

plt.figure(figsize=(9, 5))
plt.title(f'{env_name} with {run_num}th run')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.plot(x, return_history, label='Episode Return', c='pink')
plt.plot(x, running_avg, label='Average Return', c='red')
plt.grid(linewidth = 0.3)
plt.legend()
plt.show()
