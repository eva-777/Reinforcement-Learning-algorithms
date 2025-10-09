"""
- An on-policy version of Proximal Policy Optimization (PPO) algorithm, computing advantages by GAE (Generalized Advantage Estimation).
- Implemented using Pytorch, and OpenAI Gym environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Normal, MultivariateNormal, Categorical


####################### Set device #######################
print("============================================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device set to : " + str(torch.cuda.get_device_name(device)))
print("============================================================================================")

#################### Experience replay ####################
class ReplayBuffer:
    """
        Usage: store agent's memory
        Args:
            - batch_size: size of memory for each gradient descent
    """
    def __init__(self, mini_batch_size):
        self.mini_batch_size = mini_batch_size

        self.state_list = []
        self.action_list = []
        self.logprob_list = []
        self.reward_list = []
        self.value_list = []
        self.terminate_list = []
        self.truncate_list = []

    def add_memory(self, state, action, logprob, reward, value, terminated, truncated):
        self.state_list.append(state)
        self.action_list.append(action)
        self.logprob_list.append(logprob)
        self.reward_list.append(reward)
        self.value_list.append(value)
        self.terminate_list.append(terminated)
        self.truncate_list.append(truncated)

    def clear_memory(self):
        self.state_list.clear()
        self.action_list.clear()
        self.logprob_list.clear()
        self.reward_list.clear()
        self.value_list.clear()
        self.terminate_list.clear()
        self.truncate_list.clear()

    def generate_batches(self):
        """
            Usage: randomly sample multiple batches through 'random shuffle'
            Return -> batch index
        """
        n_states = len(self.state_list)
        batch_start = np.arange(0, n_states, self.mini_batch_size)
        batch_index = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(batch_index)
        batches = [batch_index[i:i+self.mini_batch_size] for i in batch_start]

        return  batches
    
########################## Actor ##########################

# Trick 8: orthogonal initialization
def orthogonal_init(use_orthogonal, layer, gain=1, bias_const=1e-6):
    """
        Args: 'gain' is to adjust the absolute value of the weight matrix. 
    """
    if use_orthogonal:
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    else:
        pass
    return layer

class Actor_Gaussian(nn.Module):
    def __init__(self, use_orthogonal, is_continuous, state_dim, action_dim, action_high, hidden_dim=64):
        super().__init__()
        
        self.is_continuous = is_continuous
        self.action_max = action_high
        self.logstd_max = 0  # todo the upper limit should gradually decay, with training
        self.logstd_min = -5.0 
        
        if is_continuous:
            self.mean_layer = nn.Sequential(
                orthogonal_init(use_orthogonal, nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, action_dim), 0.01)
            )
            self.logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Sequential(
                orthogonal_init(use_orthogonal, nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, action_dim)),
                nn.Softmax(dim=-1)
            )

    def forward(self, state):
        if self.is_continuous:
            mean = torch.tanh(self.mean_layer(state)) * self.action_max  # [-1,1] -> [-max, max]
            logstd = torch.clamp(self.logstd, self.logstd_min, self.logstd_max)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)  # [-5,0] -> [0.007, 1]
            dist = Normal(mean, std)
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
        
        return dist

    def act(self, state):
        """
            Usage: for exploration during interaction, **without considering gradient**
        """  
        dist = self.forward(state)
        action = dist.sample()
        if self.is_continuous:
            # NOTE: suppose the multi-dimensional actions are independent of each other, then the probability of joint action can be calculated by the product of independent probabilities.  -> sum(logprob)
            action = torch.clamp(action, -self.action_max, self.action_max)  # Clamp action
            logprob = dist.log_prob(action)
            return action.detach(), logprob.detach().sum()
        else:
            logprob = dist.log_prob(action)
            return action.detach(), logprob.detach()  # todo multi-discrete action space 

    def get_prob(self, state, action):
        """
            Usage: compute log_prob and entropy of action, during update, **with considering gradient**
        """  
        dist = self.forward(state)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        if self.is_continuous:
            return logprob.sum(1).flatten(), dist_entropy.sum(1).flatten()
        else:
            return logprob.flatten(), dist_entropy.flatten()


######################### Critic #########################
class Critic(nn.Module):
    def __init__(self, use_orthogonal, state_dim, hidden_dim=64):
        super().__init__()
        
        self.critic = nn.Sequential(
            orthogonal_init(use_orthogonal, nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, 1))
        )

    def forward(self, state):
        value = self.critic(state)

        return value


######################### PPO agent #########################
class Agent:
    def __init__(self, use_orthogonal, use_grad_clip, use_value_clip, is_continuous, state_dim, action_dim, gamma, lamda, eps_clip, lr_actor, lr_critic, k_epoch, batch_size, max_timesteps, action_high):
        """
            NOTE: when is continuous, action_high=float; when not continuous, action_high=None
        """
        self.use_grad_clip = use_grad_clip
        self.use_value_clip = use_value_clip

        self.is_continuous = is_continuous
        self.gamma = gamma
        self.lamda = lamda
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch
        self.max_timesteps = max_timesteps

        # On-policy
        self.actor = Actor_Gaussian(use_orthogonal, is_continuous, state_dim, action_dim, action_high).to(device)  # True -> orthogonal
        self.critic = Critic(use_orthogonal, state_dim).to(device)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Separate optimizer for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.replay_buffer = ReplayBuffer(batch_size)

    def take_action(self, state):
        """
            Usage: sample action during interaction and evaluate state-value, **without considering gradient**
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            value = self.critic(state)
            action, prob = self.actor.act(state)

        if self.is_continuous:
            return action.cpu().numpy().flatten(), prob.item(), value.item()  
        else:
            return action.item(), prob.item(), value.item()  # atten 多维离散动作空间的action可能需要flatten()

    def update_policy(self, last_state, current_step):

        # -------------------- Get memory from buffer -------------------- #
        reward_arr = np.array(self.replay_buffer.reward_list)
        terminated_arr = np.array(self.replay_buffer.terminate_list)
        truncated_arr = np.array(self.replay_buffer.truncate_list)
        value_arr = np.array(self.replay_buffer.value_list)

        # Compute last value
        # NOTE: if terminate[-1]==True, last state makes no sense, then last value = 0
        last_value = 0 if terminated_arr[-1] else self.critic(torch.FloatTensor(last_state).to(device)).item()
        next_value_arr = np.append(value_arr[1:], last_value)

        state = torch.FloatTensor(np.array(self.replay_buffer.state_list)).to(device)
        action = torch.FloatTensor(np.array(self.replay_buffer.action_list)).to(device)
        old_prob = torch.FloatTensor(np.array(self.replay_buffer.logprob_list)).to(device)
        old_value = torch.FloatTensor(value_arr).to(device)

        # ----------------------- Compute advantages ----------------------- #
        T = len(reward_arr)
        advantage = np.zeros(T, dtype=np.float32)
        gae = 0

        for t in reversed(range(T)):  # 反向计算
            # NOTE: MDP terminates or truncates, GAE should be re-accumulate
            delta_t = reward_arr[t] + self.gamma * next_value_arr[t] * (1-int(terminated_arr[t])) - value_arr[t] 
            done = int(terminated_arr[t] or truncated_arr[t])
            gae = delta_t + self.gamma * self.lamda * gae * (1-done)
            advantage[t] = gae

        advantage = torch.FloatTensor(advantage).to(device)
        value_target = advantage + old_value
        advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))  # Trick 1: advantage normalization

        # Optimize policy for K epochs
        for _ in range(self.k_epoch):
            batches = self.replay_buffer.generate_batches()

            for batch in batches:
                # ----------------------- Update actor ----------------------- #
                old_batch_logprob = old_prob[batch]
                
                batch_state = state[batch]
                batch_action = action[batch]
                batch_logprob, batch_entropy = self.actor.get_prob(batch_state, batch_action)

                ratio = torch.exp(batch_logprob - old_batch_logprob.detach())

                # ratio clip
                surr1 = ratio * advantage[batch]
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
                actor_loss = -torch.min(surr1, surr2).mean()
                # actor_loss = -torch.min(surr1, surr2).mean() - 0.01*batch_entropy  # Trick 5: policy entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # Trick 7: Gradient clip
                self.actor_optimizer.step()

                # ----------------------- Update critic ----------------------- #
                batch_value = torch.squeeze(self.critic(batch_state))
                old_batch_value = old_value[batch]
                batch_value_target = value_target[batch]
        
                # Trick 11: value clip
                if self.use_value_clip:
                    clipped_value = old_batch_value + torch.clamp(batch_value-old_batch_value, -0.3, 0.3)
                    loss_unclipped = nn.functional.mse_loss(batch_value, batch_value_target)
                    loss_clipped = nn.functional.mse_loss(clipped_value, batch_value_target)
                    critic_loss =  torch.max(loss_unclipped, loss_clipped)
                else:
                    critic_loss = nn.functional.mse_loss(batch_value, batch_value_target)  # NOTE: 该函数已默认执行mean()操作
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # Trick 7: Gradient clip
                self.critic_optimizer.step()
        
        # Clear buffer
        self.replay_buffer.clear_memory()

        # self.lr_linear_decay(current_step)  # Trick 6:learning rate Decay

        return actor_loss.item(),torch.mean(batch_entropy).item(), critic_loss.item()     

    def lr_linear_decay(self, current_step, start_decay=0.2, end_decay=0.8):
        decay_start = start_decay * self.max_timesteps
        decay_end = end_decay * self.max_timesteps
        if current_step <= decay_start:
            pass
        else:
            progress = (current_step - decay_start) / (decay_end - decay_start)

            lr_actor_new = max(self.lr_actor * (1 - progress), 1e-4)  # 最小衰减至1e-4
            lr_critic_new = max(self.lr_critic * (1 - progress), 2e-4)  # 最小衰减至1e-3
            for p in self.actor_optimizer.param_groups:
                p['lr'] = lr_actor_new
            for p in self.critic_optimizer.param_groups:
                p['lr'] = lr_critic_new  

    def save_model(self, checkpoint_path):
            torch.save(self.actor.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path))             
 
 