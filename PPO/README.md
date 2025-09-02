# PPO-PyTorch


## Introduction

- This code provides a single threaded PyTorch implementation of Proximal Policy Optimization (PPO) for OpenAI gym environments.
- The algorithm can be implemented based on **On-policy** or **Off-policy** version.
- The 'advantages' can be computed using **GAE (Generalized Advantage Estimate)** or **MC (Monte-Carlo) estimate**.


## Usage

- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`

#### Note

- Some hyperparameters should be tuned or changed for using under customized or complex environments.
- A thorough explaination of all the details for implementing best performing PPO can be found [here](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). 


## Package version

```
python == 3.11.11  
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
- [Github repository from nikhilbarhate99](https://github.com/nikhilbarhate99/PPO-PyTorch) 
- [Github repository philtabor](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch)


