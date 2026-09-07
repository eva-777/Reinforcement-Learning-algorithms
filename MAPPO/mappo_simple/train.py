import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from pettingzoo.mpe import simple_spread_v3

from agents import MAPPO
from utils import log_message, plot_metrics


############################## Initialize environment ##############################
num_agent = 2
is_continuous = False
env = simple_spread_v3.parallel_env(N=num_agent, continuous_actions=is_continuous)
env_name = 'simple_spread_v3'
agent_list = env.possible_agents
state_dim = env.observation_space(agent_list[0]).shape[0]
action_dim = env.action_space(agent_list[0]).n

################################ Set hyperparameters ################################
num_episode = 20000

lr_actor = 3e-4
lr_critic = 1e-3
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps_clip = 0.3

################################## Log, Save, Plot ##################################
env_dir = f'./on_policy_simple/results_{env_name}_{is_continuous}'
if not os.path.exists(env_dir):
          os.makedirs(env_dir)

### log training ###
log_dir = env_dir + '/log'
if not os.path.exists(log_dir):
          os.makedirs(log_dir) 

run_num = 0
current_files = next(os.walk(log_dir))[2]
run_num = len(current_files)

log_path = os.path.join(log_dir, f'training_log_{run_num}.txt')
log_message(log_path, f"Env='{env_name}', num_actor={num_agent}, Record=last50\n", timestamp=True)
log_message(log_path, 'Ep, Total_reward, Ep_Length, Actor_Loss, Critic_Loss, Avg_Entropy')

### save weights ###
chkpt_dir = env_dir + '/weight'
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

### plot result ###
plot_dir = env_dir + "/plot"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir) 

############################### Printing training info ###############################
print("============================================================================================")
print("Hyperparameters of MAPPO:")
# print(f"environment: {env_name},  is_continuous: {is_continuous}")
# print(f"total_episodes: {num_episode},  episode_length: {EPI_LEN}")
# print(f"k_epochs: {K_EPOCHS},  batch_size: {BATCH_SIZE},  update_interval: {UPDATE_INTERVAL}")
print(f"actor_lr: {lr_actor},  critic_lr: {lr_critic}")
print(f"gamma: {gamma},  lambda: {lmbda},  eps_clip: {eps_clip}")
print("============================================================================================")


################################ Training procedure ################################
# 创建MAPPO智能体（共有num_agent个actor, 一个共享critic）
mappo = MAPPO(num_agent, state_dim, hidden_dim, action_dim, lr_actor, lr_critic, lmbda, eps_clip, gamma)

# 用于统计指标的列表
total_rewards_per_episode = []
episode_lengths = []
policy_losses = []
value_losses = []
entropies = []

# 每50个episode的平均值列表
avg_total_rewards_per_50 = []
avg_episode_length_per_50 = []
avg_policy_loss_per_50 = []
avg_value_loss_per_50 = []
avg_entropy_per_50 = []

start_time = datetime.now().replace(microsecond=0)

# ----------------------- training loop ----------------------- #
for ep_i in range(1, num_episode+1):
    
    # 初始化Trajectory
    reset_result = env.reset()
    states_dict, infos = reset_result # states: {'agent_0': array, ...} -> dict 

    terminal = False
    episode_reward = 0.0
    timestep = 0

    while not terminal:
        timestep += 1
        # atten: 针对 PettingZoo API 返回格式, 做出修改
        # 将state_dict转变为state_list
        states_list = [states_dict[agent] for agent in agent_list]
        
        # MAPPO中，每个智能体仍独立选择动作，但critic共享
        actions, action_probs = mappo.take_action(states_list)

        # 将action_list转变为action_dict
        action_dict = {agent_list[i]: actions[i] for i in range(num_agent)}
        next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(action_dict)  # _: infos
        
        # 合并 terminations 和 truncations 为dones
        dones_dict = {agent: terminations_dict[agent] or truncations_dict[agent] for agent in agent_list}

        # 累计总奖励
        step_reward = sum(rewards_dict.values())  # 所有智能体奖励之和
        episode_reward += step_reward

        # 存储transition
        mappo.buffers.add_memory(agent_list, states_dict, actions, next_states_dict, rewards_dict, dones_dict, action_probs)

        states_dict = next_states_dict
        terminal = all(dones_dict.values())

    # 使用MAPPO更新参数
    actor_loss, critic_loss, entropy = mappo.update()

    # 记录指标
    total_rewards_per_episode.append(episode_reward)
    episode_lengths.append(timestep)
    policy_losses.append(actor_loss)
    value_losses.append(critic_loss)
    entropies.append(entropy)

    # 每50个episode统计一次平均值并记录、打印、绘图
    if ep_i % 50 == 0:
        avg_reward_50 = np.mean(total_rewards_per_episode[-50:])
        avg_length_50 = np.mean(episode_lengths[-50:])
        avg_policy_loss_50 = np.mean(policy_losses[-50:])
        avg_value_loss_50 = np.mean(value_losses[-50:])
        avg_entropy_50 = np.mean(entropies[-50:])

        avg_total_rewards_per_50.append(avg_reward_50)
        avg_episode_length_per_50.append(avg_length_50)
        avg_policy_loss_per_50.append(avg_policy_loss_50)
        avg_value_loss_per_50.append(avg_value_loss_50)
        avg_entropy_per_50.append(avg_entropy_50)
        
        # 记录
        log_message(log_path, f"{ep_i}: "  # episode
                              f"{avg_reward_50:.3f}, "  # Average Total reward
                              f"{avg_length_50:.3f}, "  # Average Episode length
                              f"{avg_policy_loss_50:.3f}, "  # Average Policy loss
                              f"{avg_value_loss_50:.3f}, "  # Average Value loss
                              f"{avg_entropy_50:.3f}")  # Average Entropy
        
        # 打印
        print(f"episode {ep_i}, total_reward {avg_reward_50:.3f}, policy_loss {avg_policy_loss_50:.3f}, value_loss {avg_value_loss_50:.3f}, entropy {avg_entropy_50:.3f}")

        # 创建指标字典
        metrics_dict = {
            "Average_Total_Reward": avg_total_rewards_per_50,
            "Average_Episode_Length": avg_episode_length_per_50,
            "Average_Policy_Loss": avg_policy_loss_per_50,
            "Average_Value_Loss": avg_value_loss_per_50, 
            "Average_Entropy": avg_entropy_per_50
        }
            
        # 绘图
        plot_metrics(plot_dir, run_num, metrics_dict, ep_i)
    
     # 保存模型的权重参数
    if ep_i % 500 == 0:  # todo 保存条件优化
        mappo.save_model(run_num, chkpt_dir)
        log_message(log_path, f"Model saved at episode {ep_i}")
        
end_time = datetime.now().replace(microsecond=0)
training_time = end_time - start_time
log_message(log_path, f"\nTraining time = {training_time}")

print("============================================================================================")
print(f"Total training time: {training_time}")
print(f"Log file saved at: {log_path}")
print(f"Model saved at: {chkpt_dir}")
print("============================================================================================")
