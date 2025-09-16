"""
- An Off-policy version of Proximal Policy Optimization (PPO) algorithm,
computing advantages by GAE (Generalized Advantage Estimation).

- Implemented using Pytorch, and OpenAI Gym environment.
"""

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, Categorical
import torch.optim as optim


####################### Set device #######################
print("============================================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device set to : " + str(torch.cuda.get_device_name(device)))
print("============================================================================================")


#################### Experience replay ####################

class ReplayBuffer:
    """
        Description: 经验回放缓冲区, 用于存储agent的经验
        Args:
            batch_size: 每次采样的批次大小
    """

    def __init__(self, batch_size):

        self.state_list = []
        self.action_list = []
        self.logprob_list = []
        self.reward_list = []
        self.value_list = []
        self.done_list = []
        
        self.BATCH_SIZE = batch_size
    
    def add_memory(self, state, action, logprob, reward, value, done):

        self.state_list.append(state)
        self.action_list.append(action)
        self.logprob_list.append(logprob)
        self.reward_list.append(reward)
        self.value_list.append(value)
        self.done_list.append(done)
    
    def generate_batches(self):
        """
            Description: 依据"随机洗牌"从buffer中随机采样多批次经验
            Return -> batch idx
        """
        memo_len = len(self.state_list)

        batch_start_point = np.arange(0, memo_len, self.BATCH_SIZE)  # 将经验分成多批次
        memory_idx = np.arange(0, memo_len, dtype=np.int32)  # 经验索引
        np.random.shuffle(memory_idx)  # 打乱索引顺序

        batches = [memory_idx[i:i+self.BATCH_SIZE] for i in batch_start_point]
        
        return  batches

    def clear_memory(self):

        self.state_list.clear()
        self.action_list.clear()
        self.logprob_list.clear()
        self.reward_list.clear()
        self.value_list.clear()
        self.done_list.clear()


###################### Actor, Critic ######################
class Actor(nn.Module):
    def __init__(self, is_continuous, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        self.is_continuous = is_continuous
        self.logstd_max = 1  # todo 随着训练，logstd上限应逐渐衰减
        self.logstd_min = -5.0
        
        # 连续动作空间
        if is_continuous:
            self.feature_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()  # 双曲正切激活函数
            )
            
            self.mean_head = nn.Linear(hidden_dim, action_dim)  # when continuous: 用分布来描述动作概率，由均值和标准差定义
            self.std_head = nn.Linear(hidden_dim, action_dim)

        # 离散动作空间
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )

        self.to(device)

    def forward(self, state):
        if self.is_continuous:
            features = self.feature_net(state)
            mean = torch.tanh(self.mean_head(features)) * 2  # atten: 针对Pendulum, [-2, 2]的动作范围
            log_std = torch.clamp(self.std_head(features), min=self.logstd_min, max=self.logstd_max)
            std = torch.exp(log_std)  # NOTE: std影响探索: 越大，动作越随机; 反之，越确定
            dist = Normal(mean, std)  # todo 高维动作空间可以假设每个动作维度之间相互独立

        else:
            probs = self.actor(state)
            dist = Categorical(probs)
        
        return dist
    
    def act(self, state):
        """
            Description: 用于经验收集过程, 从分布中随机采样动作, 不考虑梯度传播
        """  

        dist = self.forward(state)
        action = dist.sample()  # when continuous: 正态分布采样
        logprob = dist.log_prob(action)

        return action.detach(), logprob.detach()
    
    def get_prob(self, state, action):
        """
            Description: 用于策略更新过程, 计算动作的概率密度和熵
        """  
        
        dist = self.forward(state)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return logprob.flatten(), dist_entropy.flatten()
    

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(device)

    def forward(self, state):
        value = self.critic(state)

        return value


######################### PPO agent #########################
class Agent:
    def __init__(self, is_continuous, state_dim, action_dim, GAMMA, LAMBDA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCH, BATCH_SIZE):
        
        # Initialize actor and critic networks
        self.is_continuous = is_continuous
        
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.EPSILON_CLIP = EPS_CLIP
        self.K_EPOCH = K_EPOCH

        # off-policy
        self.actor = Actor(is_continuous, state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_old = Actor(is_continuous, state_dim, action_dim)
        self.critic_old = Critic(state_dim)

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
    
        # separate optimizer for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.replay_buffer = ReplayBuffer(BATCH_SIZE)
    
    def take_action(self, state):
        """
            Description: 用于经验收集过程, 旧网络从分布中随机采样动作, 并评估状态价值
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device) 
            action, logprob = self.actor_old.act(state)  # atten 理论上应对action进行clamp, 但gym时间已内含clamp操作
            value = self.critic_old(state)

        if self.is_continuous:
            # NOTE: 不能直接将gpu上的tensor转换为numpy, 需先转换至cpu上
            return action.cpu().numpy().flatten(), logprob.item(), value.item()
        else:
            return action.item(), logprob.item(), value.item()  # atten 多维离散动作空间的action可能需要flatten()

    def update_policy(self, last_state):
        """
            Description: 更新actor和critic网络的策略
        """
        # ----------------------- 提取buffer ----------------------- #
        rewards_arr = np.array(self.replay_buffer.reward_list)
        dones_arr = np.array(self.replay_buffer.done_list)
        values_arr = np.array(self.replay_buffer.value_list)
 
        # 添加最后一个状态价值
        if dones_arr[-1]:
            last_value = 0
        else:
            last_value = self.critic(torch.FloatTensor(last_state).to(device)).item()
        next_values_arr = np.append(values_arr[1:], last_value)

        states = torch.FloatTensor(np.array(self.replay_buffer.state_list)).detach().to(device)  
        actions = torch.FloatTensor(np.array(self.replay_buffer.action_list)).detach().to(device)
        old_logprobs = torch.FloatTensor(np.array(self.replay_buffer.logprob_list)).detach().to(device)
        old_values = torch.FloatTensor(np.array(self.replay_buffer.value_list)).detach().to(device)  

        # ----------------------- 计算Advantages ----------------------- #
        T = len(rewards_arr)
        advantages = np.zeros(T)
        gae_adv = 0

        for t in reversed(range(T)):  # 反向计算
            delta_t = rewards_arr[t] + self.GAMMA*next_values_arr[t]*(1-int(dones_arr[t])) - values_arr[t]
            gae_adv = delta_t + self.GAMMA*self.LAMBDA*gae_adv * (1-int(dones_arr[t])) 
            advantages[t] = gae_adv
        
        advantages = torch.FloatTensor(advantages).detach().to(device)
        
        # old_returns -> value target
        old_returns = advantages + old_values

        # optimize policy for K epochs
        for _ in range(self.K_EPOCH):
            # random sampling
            batches = self.replay_buffer.generate_batches()

            # 遍历batch
            for batch in batches:
                # ----------------------- 更新actor ----------------------- #
                # 计算概率比率. 注意: 此处计算概率密度即可
                old_batch_logprobs = old_logprobs[batch]

                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_logprobs, batch_entropy = self.actor.get_prob(batch_states, batch_actions)
                
                ratio = torch.exp(batch_logprobs - old_batch_logprobs.detach())
                rA = ratio*advantages[batch]
                rA_clip = torch.clamp(ratio, 1-self.EPSILON_CLIP, 1+self.EPSILON_CLIP) * advantages[batch]

                # actor损失函数: loss_clip
                actor_loss = -torch.min(rA, rA_clip).mean()
                
                # ----------------------- 更新critic ----------------------- #
                # critic损失函数: loss_vf
                batch_values = self.critic(batch_states).flatten()
                old_batch_returns = old_returns[batch]

                critic_loss = nn.functional.mse_loss(batch_values, old_batch_returns)  # NOTE: 该函数默认执行mean()操作

                # 总梯度
                total_loss = actor_loss + 0.5*critic_loss
                # total_loss = actor_loss + 0.5*critic_loss - 0.01*batch_entropy.mean()  # 考虑dist_entropy

                # 梯度更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # 旧策略网络同步新网络
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # 清空经验回放缓冲区
        self.replay_buffer.clear_memory()

        return actor_loss.item(),torch.mean(batch_entropy).item(), critic_loss.item()


    def save_model(self, checkpoint_path):
            torch.save(self.actor.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path))       
        
    