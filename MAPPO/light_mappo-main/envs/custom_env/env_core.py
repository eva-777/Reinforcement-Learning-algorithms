import numpy as np


class EnvCore(object):
    
    def __init__(self):
        self.agent_num = 2      # 智能体个数        # set the number of agents(aircrafts)
        self.obs_dim = 14       # 智能体观测维度    # set the observation dimension of agents
        self.action_dim = 5     # 智能体动作维度    # set the action dimension of agents

    def reset(self):
        """
        - agent_num设定为2个智能体时, 返回值为一个list, 每个list里面为一个shape = (obs_dim, )的观测数据
        - When agent_num = 2, the return value is a list, each list contains a observation data with shape = (obs_dim, )
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        - agent_num设定为2个智能体时, actions的输入为一个2纬的list, 每个list里面为一个shape = (action_dim, )的动作数据
        - 默认参数情况下, 输入为一个list, 里面含有两个元素, 因为动作维度为5, 所里每个元素shape = (5, )
        - When agent_num = 2, the input of actions is a 2-dimensional list, each list contains a action data with shape = (self.action_dim, ) 
        - The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
