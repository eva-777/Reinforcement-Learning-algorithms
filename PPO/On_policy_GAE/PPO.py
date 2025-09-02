"""
- An on-policy version of Proximal Policy Optimization (PPO) algorithm,
computing advantages by GAE (Generalized Advantage Estimation).
- Implemented using Pytorch, and OpenAI Gym environment.
- Referred to 'https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch'
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal, Categorical


####################### Set device #######################
print("============================================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device set to : " + str(torch.cuda.get_device_name(device)))
print("============================================================================================")


#################### Experience replay ####################
class ReplayBuffer:
    def __init__(self, batch_size):
        
        self.state_list = []
        self.action_list = []
        self.logprob_list = []
        self.reward_list = []
        self.value_list = []
        self.done_list = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.state_list)
        batch_start = np.arange(0, n_states, self.batch_size)
        index = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(index)
        batches = [index[i:i+self.batch_size] for i in batch_start]

        return  batches

    def add_memory(self, state, action, logprob, reward, value, done):
        self.state_list.append(state)
        self.action_list.append(action)
        self.logprob_list.append(logprob)
        self.reward_list.append(reward)
        self.value_list.append(value)
        self.done_list.append(done)

    def clear_memory(self):
        self.state_list = []
        self.action_list = []
        self.logprob_list = []
        self.reward_list = []
        self.value_list = []
        self.done_list = []
    

########################## Actor ##########################
class Actor(nn.Module):
    def __init__(self, is_continuous, state_dim, action_dim, lr_actor, hidden_dim=64):
        super(Actor, self).__init__()

        self.is_continuous = is_continuous
        self.logstd_max = 1.1  # todo 随着训练，logstd上限逐渐衰减
        self.logstd_min = -5.0 

        if is_continuous:
            self.feature_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.std_layer = nn.Linear(hidden_dim, action_dim)
        
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
            std = torch.exp(log_std)
            dist = Normal(mean, std)   # todo 多维
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
        
        return dist
    
    def act(self, state):

        dist = self.forward(state)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        return action, logprob

    def get_prob(self, state, action):

        dist = self.forward(state)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return logprob.flatten(), dist_entropy.flatten()


######################### Critic #########################
class Critic(nn.Module):
    def __init__(self, state_dim, lr_critic, hidden_dim=64):
        super(Critic, self).__init__()

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
        
        self.is_continuous = is_continuous
        
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.EPS_CLIP = EPS_CLIP
        self.K_EPOCH = K_EPOCH

        self.actor = Actor(is_continuous, state_dim, action_dim, LR_ACTOR)
        self.critic = Critic(state_dim, LR_CRITIC)

        # Separate optimizer for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Shared optimizer for both actor and critic
        self.shared_optimizer = torch.optim.Adam([
            {'params': self.actor.parameters() , 'lr': LR_ACTOR},
            {'params': self.critic.parameters(), 'lr': LR_CRITIC}])
        
        self.replay_buffer = ReplayBuffer(BATCH_SIZE)

    def select_action(self, state):
        
        state = torch.FloatTensor(state).to(device)
        value = self.critic(state)
        action, prob = self.actor.act(state)

        if self.is_continuous:
            return action.cpu().numpy().flatten(), prob.item(), value.item()  # todo 多维动作空间的prob是否需要flatten()?
            
        else:
            return action.item(), prob.item(), value.item()

    def update_policy(self, last_state):
        
        value_arr = np.array(self.replay_buffer.value_list)
        reward_arr = np.array(self.replay_buffer.reward_list)
        done_arr = np.array(self.replay_buffer.done_list)

        # 添加最后一个状态价值
        if done_arr[-1]:
            last_value = 0
        else:
            last_value = self.critic(torch.FloatTensor(last_state).to(device)).item()
        value_arr_ = np.append(value_arr, last_value)

        # 反向计算Advantages
        T = len(reward_arr)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0

        for t in reversed(range(T)):
            delta_t = reward_arr[t] + self.GAMMA*value_arr_[t+1]*(1-int(done_arr[t])) - value_arr_[t]
            gae = delta_t + self.GAMMA*self.LAMBDA*gae*(1-int(done_arr[t]))
            advantages[t] = gae

        advantages = torch.FloatTensor(advantages).to(device)
        old_states = torch.FloatTensor(np.array(self.replay_buffer.state_list)).to(device)
        old_actions = torch.FloatTensor(np.array(self.replay_buffer.action_list)).to(device)
        old_probs = torch.FloatTensor(np.array(self.replay_buffer.logprob_list)).to(device)
        old_values = torch.FloatTensor(value_arr).to(device)
        old_returns = advantages + old_values

        # Optimize policy for K epochs
        for _ in range(self.K_EPOCH):
            batches = self.replay_buffer.generate_batches()

            for batch in batches:
                # 计算概率比
                old_batch_probs = old_probs[batch]
                
                batch_states = old_states[batch]
                batch_actions = old_actions[batch]
                batch_probs, batch_entropy = self.actor.get_prob(batch_states, batch_actions)

                ratio = batch_probs.exp() / old_batch_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()

                # actor损失函数: loss_clip
                rA = ratio * advantages[batch]
                rA_clip = torch.clamp(ratio, 1-self.EPS_CLIP, 1+self.EPS_CLIP) * advantages[batch]
                actor_loss = -torch.min(rA, rA_clip).mean()

                # critic损失函数: loss_vf
                batch_values = self.critic(batch_states)
                batch_values = torch.squeeze(batch_values)
                old_batch_returns = old_returns[batch]
                # critic_loss = ((batch_returns-batch_values)**2).mean()
                critic_loss = nn.functional.mse_loss(batch_values, old_batch_returns)  # 该函数默认返回mean()

                # 总梯度
                total_loss = actor_loss + 0.5*critic_loss
                # total_loss = actor_loss + 0.5*critic_loss - 0.01*batch_entropy.mean()

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
        
        # 清空缓冲区
        self.replay_buffer.clear_memory()     

    def save_model(self, checkpoint_path):
            torch.save(self.actor.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path))             
 
 