"""
- An on-policy version of Proximal Policy Optimization (PPO) algorithm for continuous action space.
- Implemented using Pytorch, and Gymnasium env.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Normal

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
            - mini_batch_size: size of memory for each gradient descent
    """
    def __init__(self, state_dim, action_dim, batch_size, mini_batch):   
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        # for interaction
        self.state_array = np.zeros((batch_size, state_dim), dtype=np.float32)
        self.action_array = np.zeros((batch_size, action_dim), dtype=np.float32)
        self.logprob_array = np.zeros((batch_size, 1), dtype=np.float32)
        self.reward_array = np.zeros((batch_size, 1), dtype=np.float32)
        self.value_array = np.zeros((batch_size, 1), dtype=np.float32)
        self.terminal_array = np.zeros((batch_size, 1), dtype=np.float32)
        self.done_array = np.zeros((batch_size, 1), dtype=np.float32)
        self.count = 0
        # for update
        self.next_value_array = np.zeros((batch_size, 1), dtype=np.float32)
        self.advantage_array = np.zeros((batch_size, 1), dtype=np.float32)

    def add_memory(self, state, action, logprob, reward, value, terminated, done):
        self.state_array[self.count] = state
        self.action_array[self.count] = action
        self.logprob_array[self.count] = logprob
        self.reward_array[self.count] = reward
        self.value_array[self.count] = value
        self.terminal_array[self.count] = terminated
        self.done_array[self.count] = done
        self.count += 1

    def generate_batches(self):
        """
            Usage: randomly sample multiple batches through 'random shuffle'
            Return -> batch index
        """
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        np.random.shuffle(batch_index)
        batches = [batch_index[i:i+self.mini_batch] for i in np.arange(0, self.batch_size, self.mini_batch)]

        return  batches
    
########################## Actor ##########################
# Trick 8: orthogonal initialization
def orthogonal_init(use_orthogonal, layer, gain=1, bias_const=1e-6):
    """
        Args: 'gain' is a scale factor to adjust the absolute value of the weight matrix.
               gain=1 for input and hidden layer; gain=0.01~0.1 for output layer (actor).
        NOTE: 较小的gain使得网络初期输出的mean较小, 有效限制action幅度, 避免动作空间不稳定或超出范围
    """
    if use_orthogonal:
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    else:
        pass
    return layer

class Actor_Gaussian(nn.Module):
    def __init__(self, use_orthogonal, state_dim, hidden_dim, action_dim, action_max):
        super().__init__()
        
        self.action_max = action_max
        self.logstd_max = 0.5
        self.logstd_min = -5 
        
        self.mean_layer = nn.Sequential(
            orthogonal_init(use_orthogonal, nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(use_orthogonal, nn.Linear(hidden_dim, action_dim), 0.01)
        )
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def mean(self, state):
        mean = self.action_max * torch.tanh(self.mean_layer(state))  # [-1,1] -> [-max, max]
        return mean

    def get_dist(self, state):
        mean = self.mean(state)
        logstd = torch.clamp(self.logstd, self.logstd_min, self.logstd_max)  # logstd clamp
        logstd = logstd.expand_as(mean)  # make the 'logstd' have the same dimension as 'mean'
        std = torch.exp(logstd)  # logstd: -5 ~ 0 -> std: 0.007 ~ 1
        dist = Normal(mean, std)

        return dist

    def get_logprob(self, state, action):
        """
            Usage: for update actor, **with considering gradient**
            state: [mini_batch, state_dim]; action: [mini_batch, action_dim]
        """  
        dist = self.get_dist(state)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return logprob.sum(-1, keepdim=True), dist_entropy.sum(-1, keepdim=True)
    

class Actor_Beta(nn.Module):
    def __init__(self, use_orthogonal, state_dim, hidden_dim, action_dim, action_max):
        super().__init__()
        

######################### Critic #########################
class Critic(nn.Module):
    def __init__(self, use_orthogonal, state_dim, hidden_dim):
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
class PPO_continuous:
    def __init__(self, args):

        # Get arguments from args
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.action_max = args.action_max

        self.max_timesteps = getattr(args, 'max_timesteps', 0)
        self.K_epochs = getattr(args, 'K_epochs', 0)
        self.batch_size = getattr(args, 'batch_size', 0)
        self.mini_batch = getattr(args, 'mini_batch', 0)
        self.gamma = getattr(args, 'gamma', 0)
        self.lamda = getattr(args, 'lamda', 0)
        self.eps_clip = getattr(args, 'eps_clip', 0)
        self.lr_actor = getattr(args, 'lr_actor', 0)
        self.lr_critic = getattr(args, 'lr_critic', 0)
        self.lr_actor_decay = getattr(args, 'lr_actor_decay', 0)
        self.lr_critic_decay = getattr(args, 'lr_critic_decay', 0)
        
        self.use_orthogonal = getattr(args, 'use_orthogonal', False)
        self.entropy_coef = getattr(args, 'entropy_coef', 0)
        self.entropy_decay = getattr(args, 'entropy_decay', 0)
        self.use_adv_norm = getattr(args, 'use_adv_norm', False)
        self.use_grad_clip = getattr(args, 'use_grad_clip', False)
        self.use_value_clip = getattr(args, 'use_value_clip', False)
        self.use_lr_decay = getattr(args, 'use_lr_decay', False)

        # construct on-policy actor-critic network
        self.actor = Actor_Gaussian(self.use_orthogonal, 
                                    self.state_dim, self.hidden_dim, self.action_dim, self.action_max).to(device)
        self.critic = Critic(self.use_orthogonal, self.state_dim, self.hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=1e-6)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-6)

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, self.batch_size, self.mini_batch)

    def take_action(self, state, deterministic):
        """
            Usage: get action and state value, **without considering gradient**
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, state_dim]
            value = self.critic(state)                                # [1, 1]
            # when evaluate, take 'mean' as max prob action
            if deterministic:
                action = self.actor.mean(state)
                return action.cpu().numpy().flatten(), None, None
            # when train, explorate action space
            else:
                dist = self.actor.get_dist(state)
                action = dist.sample()
                action = action.clamp(-self.action_max, self.action_max)  # clamp action 
                # NOTE: suppose multi-dimensional actions are independent of each other, then the joint action probability can be the product of independent probabilities.  -> sum(logprob)
                logprob = dist.log_prob(action)
            
                return action.cpu().numpy().flatten(), logprob.sum(-1).item(), value.item()  
 
    def update_policy(self, last_state, current_step):
        # exploration decay
        if self.entropy_coef > 1e-4:
            self.entropy_coef *= self.entropy_decay
        else:
            self.entropy_coef = 0

        # -------------------- Get memory from buffer -------------------- #
        state = torch.FloatTensor(self.buffer.state_array).to(device)          # [T, state_dim]
        action = torch.FloatTensor(self.buffer.action_array).to(device)        # [T, action_dim]
        logprob = torch.FloatTensor(self.buffer.logprob_array).to(device)      # [T, 1]
        reward = self.buffer.reward_array            # (T, 1)
        value = self.buffer.value_array              # (T, 1)
        terminal = self.buffer.terminal_array        # (T, 1)
        done = self.buffer.done_array                # (T, 1)
        
        # construct next_value_arr
        # NOTE: if terminate[-1]==True, last_state makes no sense, then last_value = 0
        last_value = 0 if terminal[-1] else self.critic(torch.FloatTensor(last_state).to(device)).item()
        self.buffer.next_value_array[0:-1] = value[1:]
        self.buffer.next_value_array[-1] = last_value
        next_value = self.buffer.next_value_array    # (T, 1)

        # ---------------------- Calculate advantages ---------------------- #
        # NOTE: Use GAE to calculate q/advantage of given memory. If MDP terminates or truncates, GAE should be re-accumulated.
        gae = 0
        for t in reversed(range(self.batch_size)):  
            delta_t = reward[t] + (1.0-terminal[t]) * self.gamma * next_value[t] - value[t] 
            gae = delta_t + (1.0-done[t]) * self.gamma * self.lamda * gae
            self.buffer.advantage_array[t] = gae

        advantage = torch.FloatTensor(self.buffer.advantage_array).to(device)    # [T, 1]
        value = torch.FloatTensor(value).to(device)                              # [T, 1]
        value_target = advantage + value  # GAE return

        # Trick 1: advantage normalization
        if self.use_adv_norm:
            advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))

        # update agent for K epochs
        for _ in range(self.K_epochs):
            batches = self.buffer.generate_batches()

            for batch in batches:
                # ----------------------- Update actor ----------------------- #
                batch_state = state[batch]
                batch_action = action[batch]
                batch_advantage = advantage[batch]
                batch_logprob = logprob[batch]
                batch_logprob_new, batch_entropy = self.actor.get_logprob(batch_state, batch_action)
            
                # ratio clip
                ratio = torch.exp(batch_logprob_new - batch_logprob)
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * batch_advantage
                
                # Trick 5: entropy loss
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * batch_entropy.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Trick 7: Gradient clip
                if self.use_grad_clip:  
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.actor_optimizer.step()

                # ----------------------- Update critic ----------------------- #
                batch_value = value[batch]
                batch_value_target = value_target[batch]
                batch_value_pred = self.critic(batch_state)

                # Trick 11: value clip
                if self.use_value_clip:  
                    clipped_value = batch_value + torch.clamp(batch_value_pred-batch_value, -0.3, 0.3)
                    loss_unclipped = nn.functional.mse_loss(batch_value_pred, batch_value_target)
                    loss_clipped = nn.functional.mse_loss(clipped_value, batch_value_target)
                    critic_loss =  torch.max(loss_unclipped, loss_clipped)
                else:
                    critic_loss = nn.functional.mse_loss(batch_value_pred, batch_value_target)  # NOTE: mse_loss()已默认执行mean()操作
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Trick 7: Gradient clip
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.critic_optimizer.step()
        
        # reset buffer count
        self.buffer.count = 0

        # Trick 6:learning rate Decay
        if self.use_lr_decay:  
            self.lr_linear_decay(current_step)

        return actor_loss.item(), torch.mean(batch_entropy).item(), critic_loss.item()     

    def lr_linear_decay(self, current_step, start_decay=0.2, end_decay=0.8):
        decay_start = start_decay * self.max_timesteps
        decay_end = end_decay * self.max_timesteps

        if current_step > decay_start:
            progress = (current_step - decay_start) / (decay_end - decay_start)

            lr_actor_new = max(self.lr_actor * (1 - progress), self.lr_actor_decay)
            lr_critic_new = max(self.lr_critic * (1 - progress), self.lr_critic_decay)
            for p in self.actor_optimizer.param_groups:
                p['lr'] = lr_actor_new
            for p in self.critic_optimizer.param_groups:
                p['lr'] = lr_critic_new  

    def save_model(self, checkpoint_path):
            torch.save(self.actor.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path))             
 