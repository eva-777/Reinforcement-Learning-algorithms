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

        self.state_list = []
        self.action_list = []
        self.logprob_list = []
        self.reward_list = []
        self.value_list = []
        self.done_list = []


###################### Actor, Critic ######################
class Actor(nn.Module):
    def __init__(self, is_continuous, state_dim, action_dim, lr_actor, hidden_dim=64):
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
            
            self.mean_layer = nn.Linear(hidden_dim, action_dim)  # 连续动作空间中，用分布来表示动作概率，分布由均值和标准差定义
            self.std_layer = nn.Linear(hidden_dim, action_dim)

        # 离散动作空间
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )

        self.to(device)

    def forward(self, state):
        if self.is_continuous:
            features = self.feature_net(state)
            mean = torch.tanh(self.mean_layer(features))
            log_std = torch.clamp(self.std_layer(features), min=self.logstd_min, max=self.logstd_max)
            std = torch.exp(log_std) # atten std影响探索: 越大，动作越随机; 反之，越确定
            dist = Normal(mean, std)

        else:
            probs = self.actor(state)
            dist = Categorical(probs)
        
        return dist
    
    def act(self, state):
        """
            Description: 用于经验收集过程, 从分布中随机采样动作
        """  

        dist = self.forward(state)
        action = dist.sample()  # 正太分布: 68%概率落在[μ-σ, μ+σ], ...
        logprob = dist.log_prob(action)

        return action.detach(), logprob.detach()
    
    def get_prob(self, state, action):
        """
            Description: 用于策略更新过程, 计算动作的概率密度
        """  
        
        dist = self.forward(state)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return logprob.flatten(), dist_entropy.flatten()
    

class Critic(nn.Module):
    def __init__(self, state_dim, lr_critic, hidden_dim=64):
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
    """
        input: state (observation)
        output: action
    """
    def __init__(self, is_continuous, state_dim, action_dim, GAMMA, LAMBDA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCH, BATCH_SIZE) -> None:
        # Initialize actor and critic networks
        self.is_continuous = is_continuous
        
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.EPSILON_CLIP = EPS_CLIP
        self.K_EPOCH = K_EPOCH

        self.actor = Actor(is_continuous, state_dim, action_dim, LR_ACTOR)
        self.critic = Critic(state_dim, LR_CRITIC)
        self.actor_old = Actor(is_continuous, state_dim, action_dim, LR_ACTOR)
        self.critic_old = Critic(state_dim, LR_CRITIC)

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
    
        # Separate optimizer for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Shared optimizer for both actor and critic
        self.shared_optimizer = torch.optim.Adam([
            {'params': self.actor.parameters() , 'lr': LR_ACTOR},
            {'params': self.critic.parameters(), 'lr': LR_CRITIC}])
        
        self.replay_buffer = ReplayBuffer(BATCH_SIZE)
    
    def select_action(self, state):
        """
            Description: 用于经验收集过程, 旧网络从分布中随机采样动作, 并评估状态价值
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)  # 转换为Tensor并移动到设备上
            action, logprob = self.actor_old.act(state)  # 从actor网络中获取动作
            value = self.critic_old(state)  # 从critic网络中获取状态价值

        if self.is_continuous:
            return action.cpu().numpy().flatten(), logprob.item(), value.item() # todo 多维连续动作空间的prob是否需要flatten()?
        else:
            return action.item(), logprob.item(), value.item()  # todo 多维离散动作空间的action是否需要flatten()?

    def update_policy(self, last_state):
        """
            Description: 更新actor和critic网络的策略
        """
        # 注意: 不能直接将gpu上的tensor转换为numpy, 可以转换cpu上的
        reward_arr = np.array(self.replay_buffer.reward_list)
        done_arr = np.array(self.replay_buffer.done_list)
        value_arr = np.array(self.replay_buffer.value_list)
        
        # 添加最后一个状态价值
        if done_arr[-1]:
            last_value = 0
        else:
            last_value = self.critic(torch.FloatTensor(last_state).to(device)).item()
        value_arr_ = np.append(value_arr, last_value)

        # 反向计算Advantages
        T = len(reward_arr)
        advantages = np.zeros(T)
        gae = 0

        for t in reversed(range(T-1)):
            delta_t = reward_arr[t] + self.GAMMA*value_arr_[t+1]*(1-int(done_arr[t])) - value_arr_[t]
            gae = delta_t + self.GAMMA*self.LAMBDA*gae*(1-int(done_arr[t])) 
            advantages[t] = gae

        advantages = torch.FloatTensor(advantages).detach().to(device)
        old_states = torch.FloatTensor(np.array(self.replay_buffer.state_list)).detach().to(device)  
        old_actions = torch.FloatTensor(np.array(self.replay_buffer.action_list)).detach().to(device)
        old_logprobs = torch.FloatTensor(np.array(self.replay_buffer.logprob_list)).detach().to(device)
        old_values = torch.FloatTensor(np.array(self.replay_buffer.value_list)).detach().to(device)     
        old_returns = advantages + old_values

        # Optimize policy for K epochs
        for _ in range(self.K_EPOCH):
            # 随机采样
            batches = self.replay_buffer.generate_batches()

            # 遍历batch
            for batch in batches:
                # 计算概率比率. 注意: 此处计算概率密度即可
                old_batch_probs = old_logprobs[batch]

                batch_states = old_states[batch]
                batch_actions = old_actions[batch]
                batch_probs, batch_entropy = self.actor.get_prob(batch_states, batch_actions)
                
                ratio = torch.exp(batch_probs - old_batch_probs.detach())
                
                # actor损失函数
                rA = ratio*advantages[batch]
                rA_clip = torch.clamp(ratio, 1-self.EPSILON_CLIP, 1+self.EPSILON_CLIP) * advantages[batch]

                actor_loss = -torch.min(rA, rA_clip).mean()

                # critic损失函数
                batch_values = self.critic(batch_states).flatten()
                old_batch_returns = old_returns[batch]

                critic_loss = nn.functional.mse_loss(batch_values, old_batch_returns)  # 该函数默认返回mean()

                # 总梯度
                total_loss = actor_loss + 0.5*critic_loss

                # 梯度更新
                # Separte optimizer for actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # # Shared optimizer for both actor and critic
                # self.shared_optimizer.zero_grad()
                # total_loss.backward()
                # self.shared_optimizer.step()

        # 旧策略网络同步新网络
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # 清空经验回放缓冲区
        self.replay_buffer.clear_memory()  


    def save_model(self, checkpoint_path):
            torch.save(self.actor.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path))       
        
    