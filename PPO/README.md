# PPO-PyTorch

## Introduction
- This code provides a single threaded PyTorch implementation of [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization) for OpenAI gym environments. The algorithm can be implemented for **discrete** or **continuous** action space.  


## Tricks
Here are some training tricks for stable and well-performing PPO implementation:
- 1. Random seed for torch, cudnn, numpy, env_train                 (√)
- 2. Orthogonal initialization                                      (√)
- 3. Advantages normalization                                       (√)
- 4. Gradient clip                                                  (√)
- 5. Value function clip                                            (√)
- 6. Learning rate decay                                            (√)
- 7. Stochastic for train, deterministic for eval                   (√)


#### Note
- Some hyperparameters should be tuned or changed for using under customized or complex environments.
- A thorough explaination of all the details for implementing best performing PPO can be found [here](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) and [here](https://arxiv.org/abs/2005.12729).


## Usage
- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`


## Package version

```
python == 3.11.13  
pyTorch == 2.3.1+cu121
torchvision == 0.18.1+cu121
gymnasium == 0.29.1   
pygame == 2.6.1  
box2d == 2.3.10  
```

```
numPy == 1.26.4  
matplotlib == 3.10.3 
```


## References
Here are some well-performing PPO implementation that can be referred:
- [Github from XinJingHao](https://github.com/XinJingHao/DRL-Pytorch)
- [Github from Lizhi-sjtu](https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous)
- [Github from nikhilbarhate99](https://github.com/nikhilbarhate99/PPO-PyTorch) 
- [Github from vwxyzjn](https://github.com/vwxyzjn/ppo-implementation-details)
