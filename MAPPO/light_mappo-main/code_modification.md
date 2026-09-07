### 一、``light_mappo-main`` 的问题
#### 1. distributions.py **(已修改)**  
	FixedNormal.entrop 拼写错误
#### 2. util.py **(已修改)**  
	huber loss: abs丢失
#### 3. config.py **(已修改)**  
	--share_policy 永远为 True
#### 4. act.py
	dist_entropy = torch.tensor(dist_entropy).mean(): MultiDiscrete 分支切断了 entropy 梯度，并可能造成设备不一致
#### 5. popart.py
    update 更新逻辑: old_mean, old_stddev = self.mean, self.stddev 与原版不一致
#### 6. render_mpe.py 已经无法在当前项目中运行
#### 7. shared/env_runner 的渲染路径不支持项目默认的连续动作环境
#### 8. env_runner.py **(已修改)** 
    注销 if self.all_args.save_gifs: ...


### 二、``on-policy-main`` 的问题
#### 1. separated_buffer.py **(已修改)** 
    compute_returns函数的一个分支条件不完整: 
    if self._use_popart: 
        ... \
        + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])

### 三、有差异，但暂不认定为错误
#### 1. mlp.py
    light_mappo 使用: 
        self.fc_h = ...
        self.fc2 = get_clones(self.fc_h, self._layer_N)
    on-policy 使用: 
    self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])