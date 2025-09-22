# PPO-PyTorch

## Introduction
- This code provides a single threaded PyTorch implementation of Proximal Policy Optimization (PPO) for OpenAI gym environments.
- The algorithm can be implemented based on **On-policy** or **Off-policy** version.
- The 'advantages' can be computed using **GAE (Generalized Advantage Estimate)** or **MC (Monte-Carlo) estimate**.  


## Tricks
Here are some training tricks for stable and well-performing PPO implementation:
- Torch random seed
- Orthogonal initialization
- Gradient clip
- Value function clip
- Learning rate decay


#### Note
- Some hyperparameters should be tuned or changed for using under customized or complex environments.
- A thorough explaination of all the details for implementing best performing PPO can be found [here](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). 


## Usage
- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`


## Package version

```
python == 3.11.13  
pyTorch == 2.3.1+cu121  
torchvision == 0.18.1+cu121  
gym == 0.26.2  
pygame == 2.6.1  
```

```
numPy == 1.26.0  
matplotlib == 3.10.3 
```


## References
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [Github from nikhilbarhate99](https://github.com/nikhilbarhate99/PPO-PyTorch) 
- [Github from vwxyzjn](https://github.com/vwxyzjn/ppo-implementation-details)
- [Github from XinJingHao](https://github.com/XinJingHao/PPO-Continuous-Pytorch/tree/main)
- [Github from Lizhi-sjtu](https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous)