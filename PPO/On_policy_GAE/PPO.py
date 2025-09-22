"""
- An on-policy version of Proximal Policy Optimization (PPO) algorithm, computing advantages by GAE (Generalized Advantage Estimation).
- Implemented using Pytorch, and OpenAI Gym environment.
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
    """
        Description: store agent's memory
        Args:
            - batch_size: size of memory for each gradient descent
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.state_list = []
        self.action_list = []
        self.logprob_list = []
        self.reward_list = []
        self.value_list = []
        self.terminate_list = []
        self.truncate_list = []

    def generate_batches(self):
        """
            Description: randomly sample multiple batches through 'random shuffle'
            Return -> batch index
        """
        n_states = len(self.state_list)
        batch_start = np.arange(0, n_states, self.batch_size)
        batch_index = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(batch_index)
        batches = [batch_index[i:i+self.batch_size] for i in batch_start]

        return  batches

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
    
########################## Actor ##########################

# Trick 8: orthogonal initialization
def orthogonal_init(layer, std=5/3, bias_const=0.0):
    """
        Args: 'std' is to adjust the absolute value of the weight matrix. 
        When act_fuc is ReLU -> sqrt(2), Tanh -> 5/3, Linear -> 1
    """
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, use_orthogonal, is_continuous, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        self.is_continuous = is_continuous
        self.logstd_max = 1  # todo the upper limit should gradually decay, with training
        self.logstd_min = -5.0 
        
        if use_orthogonal:
            if is_continuous:
                self.feature_net = nn.Sequential(
                    orthogonal_init(nn.Linear(state_dim, hidden_dim)),
                    nn.Tanh(),
                    orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
                    nn.Tanh()
                )
                self.mean_head = orthogonal_init(nn.Linear(hidden_dim, action_dim), 0.01)
                self.std_head = orthogonal_init(nn.Linear(hidden_dim, action_dim), 0.01)
            else:
                self.actor = nn.Sequential(
                    orthogonal_init(nn.Linear(state_dim, hidden_dim)),
                    nn.Tanh(),
                    orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
                    nn.Tanh(),
                    orthogonal_init(nn.Linear(hidden_dim, action_dim)),
                    nn.Softmax(dim=-1)
                )
        else:
            if is_continuous:
                self.feature_net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh()
                )
                self.mean_head = nn.Linear(hidden_dim, action_dim)
                self.std_head = nn.Linear(hidden_dim, action_dim)
            else:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Softmax(dim=-1)
                )

    def forward(self, state):
        if self.is_continuous:
            features = self.feature_net(state)
            mean = torch.tanh(self.mean_head(features)) * 2  # atten: 针对Pendulum, [-2, 2]的动作范围
            log_std = torch.clamp(self.std_head(features), min=self.logstd_min, max=self.logstd_max)
            std = torch.exp(log_std)
            dist = Normal(mean, std)  # todo joint-action probability under multi-dimension action space
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
        
        return dist

    def act(self, state):
        """
            Description: sample action from dist during interaction process, without considering gradient
        """  
        dist = self.forward(state)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        return action, logprob

    def get_prob(self, state, action):
        """
            Description: compute log_prob and entropy of action, during update process, with considering gradient
        """  
        dist = self.forward(state)
        logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return logprob.flatten(), dist_entropy.flatten() 


######################### Critic #########################
class Critic(nn.Module):
    def __init__(self, use_orthogonal, state_dim, hidden_dim=64):
        super().__init__()
        
        if use_orthogonal:
            self.critic = nn.Sequential(
                orthogonal_init(nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                orthogonal_init(nn.Linear(hidden_dim, 1))
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, state):
        value = self.critic(state)

        return value


######################### PPO agent #########################
class Agent:
    def __init__(self, use_orthogonal, is_continuous, state_dim, action_dim, GAMMA, LAMBDA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCH, BATCH_SIZE):
        
        self.is_continuous = is_continuous
        
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.EPS_CLIP = EPS_CLIP
        self.K_EPOCH = K_EPOCH

        # On-policy 
        self.actor = Actor(use_orthogonal, is_continuous, state_dim, action_dim).to(device)  # True -> orthogonal
        self.critic = Critic(use_orthogonal, state_dim).to(device)
 
        # Separate optimizer for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.replay_buffer = ReplayBuffer(BATCH_SIZE)

    def take_action(self, state):
        """
            Description: sample action during interaction process and evaluate state-value, without considering gradient
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            value = self.critic(state)
            action, prob = self.actor.act(state)  # atten 理论上应对action进行clamp, 但gym时间已内含clamp操作

        if self.is_continuous:
            return action.cpu().numpy().flatten(), prob.item(), value.item()  # atten 多维离散动作空间的action可能需要flatten()
        else:
            return action.item(), prob.item(), value.item()

    def update_policy(self, last_state):

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
            delta_t = reward_arr[t] + self.GAMMA*next_value_arr[t]*(1-int(terminated_arr[t])) - value_arr[t] 
            done = int(terminated_arr[t] or truncated_arr[t])
            gae = delta_t + self.GAMMA*self.LAMBDA*gae*(1-done)
            advantage[t] = gae

        advantage = torch.FloatTensor(advantage).to(device)
        # advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))

        value_target = advantage + old_value

        # Optimize policy for K epochs
        for _ in range(self.K_EPOCH):
            batches = self.replay_buffer.generate_batches()

            for batch in batches:
                # ----------------------- Update actor ----------------------- #
                old_batch_logprob = old_prob[batch]
                
                batch_state = state[batch]
                batch_action = action[batch]
                batch_logprob, batch_entropy = self.actor.get_prob(batch_state, batch_action)

                ratio = torch.exp(batch_logprob - old_batch_logprob.detach())

                # loss_clip
                surr1 = ratio * advantage[batch]
                surr2 = torch.clamp(ratio, 1-self.EPS_CLIP, 1+self.EPS_CLIP) * advantage[batch]
                # actor_loss = -torch.min(surr1, surr2).mean() - 0.01*batch_entropy  # Trick 5: policy entropy
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # Trick 7: Gradient clip
                self.actor_optimizer.step()

                # ----------------------- Update critic ----------------------- #
                # loss_vf
                batch_value = torch.squeeze(self.critic(batch_state))
                batch_value_target = value_target[batch]
                critic_loss = nn.functional.mse_loss(batch_value, batch_value_target)  # NOTE: 该函数已默认执行mean()操作
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # Trick 7: Gradient clip
                self.critic_optimizer.step()
        
        # Clear buffer
        self.replay_buffer.clear_memory()

        return actor_loss.item(),torch.mean(batch_entropy).item(), critic_loss.item()     

    def save_model(self, checkpoint_path):
            torch.save(self.actor.state_dict(), checkpoint_path)

    def load_model(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path))             
 
 