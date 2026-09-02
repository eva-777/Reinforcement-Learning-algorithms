import os
import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, Categorical


################################# Set device #################################
print("============================================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device set to : " + str(torch.cuda.get_device_name(device)))
print("============================================================================================")

############################# Experiences replay #############################
class ReplayBuffer:
    def __init__(self, num_agent):
        """
           self.dicts -> List[dict]
        """

        self.num_agent = num_agent
        self.dicts = [{
        'state': [],      # 单个智能体的buffer, 后缀不加s 
        'action': [], 
        'next_state': [], 
        'reward': [], 
        'done': [], 
        'action_prob': []
        } for _ in range(num_agent)]
    
    def add_memory(self, agent_list, states_dict, actions, next_states_dict, rewards_dict, dones_dict, action_probs):
        """
            dict <- states_dict, next_states_dict, rewards_dict, dones_dict
            list <- actions, action_probs
        """
        
        for i, agent in enumerate(agent_list):
            self.dicts[i]['state'].append(states_dict[agent])
            self.dicts[i]['action'].append(actions[i])
            self.dicts[i]['next_state'].append(next_states_dict[agent])
            self.dicts[i]['reward'].append(rewards_dict[agent])
            self.dicts[i]['done'].append(dones_dict[agent])
            self.dicts[i]['action_prob'].append(action_probs[i])

    def clear_memory(self):
        for i in range(self.num_agent):
            self.dicts[i]['state'].clear()
            self.dicts[i]['action'].clear()
            self.dicts[i]['next_state'].clear()
            self.dicts[i]['reward'].clear()
            self.dicts[i]['done'].clear()
            self.dicts[i]['action_prob'].clear()

############################# Actor, Critic #############################
# 策略网络(Actor)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        probs = self.actor(state)

        return probs

# 全局价值网络(Central Critic)
class CentralValueNet(nn.Module):
    """
        Input: 所有智能体的状态拼接 (num_agent * state_dim)
        Output: 对每个智能体的价值估计 (num_agent维向量)
    """
    def __init__(self, total_state_dim, hidden_dim, num_agent):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(total_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_agent)  # 输出为每个智能体一个价值
        )

    def forward(self, state):
        values = self.critic(state)

        return values  # [batch, num_agent]


################################ MAPPO ################################

def compute_entropy(probs):
    dist = torch.distributions.Categorical(probs)
    return dist.entropy().mean().item()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    # 反向计算
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()

    return torch.tensor(advantage_list, dtype=torch.float)

class MAPPO:
    def __init__(self, num_agent, state_dim, hidden_dim, action_dim, lr_actor, lr_critic, lmbda, eps_clip, gamma):
        
        # 参数
        self.num_agent = num_agent
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip

        # Multi-actor
        self.actors = [PolicyNet(state_dim, hidden_dim, action_dim).to(device) for _ in range(num_agent)]

        # 全局critic
        self.critic = CentralValueNet(num_agent * state_dim, hidden_dim, num_agent).to(device)

        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr_actor) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr_critic)

        self.buffers = ReplayBuffer(num_agent)

    def take_action(self, states):
        # states: shape=[num_agent, state_dim] -> List[array]
        actions = []
        action_probs = []
        for i, actor in enumerate(self.actors):
            with torch.no_grad():
                state_i = torch.tensor(states[i], dtype=torch.float).to(device)
                probs_i = actor(state_i)
                action_dist_i = torch.distributions.Categorical(probs_i)
                action_i = action_dist_i.sample()
            
            actions.append(action_i.item())
            action_probs.append(probs_i.detach().cpu().numpy())

        return actions, action_probs

    def update(self):
        """
        buffer_dicts: shape [num_agent, dict] -> 1ist[dict]
            字典结构:
            state: shape [T, state_dim]
            action: shape [T]
            next state: shape [T, state dim]
            reward: shape [T]
            done: shape [T]
            action prob: shape [T, action dim]
        """
        # ----------------------- 提取memory ----------------------- #
        buffer_dicts = self.buffers.dicts

        # 假设所有智能体memory长度相同（因为同步环境步）
        T = len(buffer_dicts[0]['state'])
        # 拼接所有智能体在同一时间步t的state, 用于训练全局critic
        states_all = []
        for t in range(T):
            concat_state = []
            for i in range(self.num_agent):
                concat_state.append(buffer_dicts[i]['state'][t])
            states_all.append(np.concatenate(concat_state))

        states_all = torch.tensor(np.array(states_all), dtype=torch.float).to(device)  # shape=[T, num_agent*state_dim]
        last_states_all = torch.tensor(np.concatenate([buffer_dicts[i]['next_state'][-1] for i in range(self.num_agent)]), dtype=torch.float).to(device)  # 当前episode的最后一个状态, shape=[num_agent*state_dim]
        rewards_all = torch.tensor([ [buffer_dicts[i]['reward'][t] for i in range(self.num_agent)] 
                                     for t in range(T)], dtype=torch.float).to(device)  # shape=[T, num_agent]
        dones_all = torch.tensor([ [buffer_dicts[i]['done'][t] for i in range(self.num_agent)] 
                                   for t in range(T)], dtype=torch.float).to(device)  #  shape=[T, num_agent]

        # ----------------------- 更新全局critic ----------------------- #
        # 从critic计算state values, TD-target
        values = self.critic(states_all)  # [T, num_agent]    
        last_values = self.critic(last_states_all)
        next_values = torch.cat((values[1:], last_values.unsqueeze(0)), dim=0)  # 拼接last_values, shape=[T, num_agent]

        td_target = rewards_all + self.gamma * next_values * (1 - dones_all) #  shape=[T, num_agent]
        td_delta = td_target - values # shape=[T, num_agent]

        # critic loss: 所有智能体的均方误差平均
        critic_loss = nn.functional.mse_loss(values, td_target.detach())  # atten td_target作为逼近项, 可能会使critic收敛慢
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------- 更新每个actor ----------------------- #
        # 为每个智能体计算优势函数
        advantages = []
        for i in range(self.num_agent):
            advantage_i = compute_advantage(self.gamma, self.lmbda, td_delta[:, i])  #  shape=[T]
            advantages.append(advantage_i.to(device))  # shape=[num_agent, T]

        action_losses = []
        entropies = []

        for i in range(self.num_agent):
            # 计算概率比
            states = torch.tensor(np.array(buffer_dicts[i]['state']), dtype=torch.float).to(device)
            actions = torch.tensor(buffer_dicts[i]['action']).view(-1, 1).to(device)
            old_probs = torch.tensor(np.array(buffer_dicts[i]['action_prob']), dtype=torch.float).to(device)
            old_logprobs = torch.log(old_probs.gather(1, actions)).detach()

            current_probs = self.actors[i](states)  # [T, action_dim]
            logprobs = torch.log(current_probs.gather(1, actions))
            entropy_val = compute_entropy(current_probs)

            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages[i].unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i].unsqueeze(-1)

            # actor loss: loss_clip
            action_loss = torch.mean(-torch.min(surr1, surr2))
            
            # 梯度更新
            self.actor_optimizers[i].zero_grad()
            action_loss.backward()
            self.actor_optimizers[i].step()

            action_losses.append(action_loss.item())
            entropies.append(entropy_val)
        
        self.buffers.clear_memory()

        return np.mean(action_losses), critic_loss.item(), np.mean(entropies)
    

    def save_model(self, run_num, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f"{run_num}_actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, f"{run_num}_critic.pth"))

    def load_model(self, run_num, path):
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"{run_num}_actor_{i}.pth")
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path))
            
        critic_path = os.path.join(path, f"{run_num}_critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))