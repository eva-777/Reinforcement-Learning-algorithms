
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

################################# 辅助函数 #################################
# 日志记录函数
def log_message(log_path, message, timestamp=False):
    with open(log_path, "a") as f:
        if timestamp:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f"[{time_str}] {message}\n")
        else:
            f.write(message + "\n")

# 作图函数
def plot_metrics(plot_dir, run_num, metrics_dict, episode):
    """
    Description: 
        - 将所有指标绘制到一个图中, 并对曲线进行平滑处理, 以及添加误差带显示
    Args:
        metrics_dict: 包含所有指标数据的字典，格式为 {metric_name: values_list}
        episode: 当前的episode数
    """
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics (Up to Episode {episode})', fontsize=16)
    
    # 压平axes数组以便迭代
    axes = axes.flatten()
    
    # 为每个指标获取x轴值
    any_metric = list(metrics_dict.values())[0]
    x_values = [50 * (i + 1) for i in range(len(any_metric))]
    
    # 平滑窗口长度
    window_size = min(5, len(x_values)) if len(x_values) > 0 else 1
    
    # 在每个子图中绘制一个指标
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
          
        ax = axes[i]
        values_array = np.array(values)
        
        # 如果数据长度>窗口长度, 平滑处理
        if len(values) > window_size:
            # 计算平滑曲线
            smoothed_y = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
            
            # 计算滚动标准差用于误差带
            std_values = []
            for j in range(len(values) - window_size + 1):
                std_values.append(np.std(values_array[j:j+window_size]))
            std_values = np.array(std_values)
            
            # 调整x轴以匹配平滑后的数据长度
            smoothed_x = x_values[window_size-1:]
            
            # 绘制平滑曲线和原始散点
            ax.plot(smoothed_x, smoothed_y, '-', linewidth=2, label='Smoothed')
            ax.scatter(x_values, values, alpha=0.3, label='Original')
            
            # 添加误差带
            ax.fill_between(smoothed_x, smoothed_y-std_values, smoothed_y+std_values, 
                           alpha=0.2, label='±1 StdDev')
        else:
            # 如果数据点太少，只绘制原始数据
            ax.plot(x_values, values, 'o-', label='Data')
        
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 删除未使用的子图
    if len(metrics_dict) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, f'training_metrics_{run_num}.png'))
    plt.close(fig)