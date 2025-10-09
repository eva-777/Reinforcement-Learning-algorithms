import argparse

def main(args):

    max_timesteps = args.max_timesteps
    max_ep_len = args.epi_len
    k_epochs = args.k_epochs
    batch_size = max_ep_len
    log_freq = max_ep_len * 2
    update_interval = max_ep_len * 4
    print_freq = update_interval * 2

    gamma = args.gamma
    lamda = args.lamda  # 注意：lambda是Python关键字，需要使用getattr
    eps_clip = args.eps_clip
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic

    use_orthogonal = True

if __name__ == '__main__':
    """
    argparse:
    1. 位置参数: 必须按顺序提供,以及指定参数值, 不需要指定参数名
    2. 可选参数: --name, 表示变量是一个可选参数, 可以按任意顺序提供, 必须指定参数名
    
    """
    
    
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_timesteps", type=int, default=int(60e4), help="Maximum number of timesteps for training")
    parser.add_argument("--max_ep_len", type=int, default=400, help="Maximum length of an episode")
    parser.add_argument("--k_epochs", type=int, default=40, help="Number of epochs for policy update")
    parser.add_argument("--batch_size", type=int, default=400, help="Batch size for training")
    parser.add_argument("--log_freq", type=int, default=800, help="Frequency of logging")
    parser.add_argument("--update_interval", type=int, default=1600, help="Interval between policy updates")
    parser.add_argument("--print_freq", type=int, default=3200, help="Frequency of printing progress")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="Clipping parameter for actor log_prob")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Learning rate for actor network")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Learning rate for critic network")

    parser.add_argument("--use_orthogonal", type=bool, default=True, help="whether to use orthogonal initialization")

    args = parser.parse_args()
    main(args)