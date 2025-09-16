
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