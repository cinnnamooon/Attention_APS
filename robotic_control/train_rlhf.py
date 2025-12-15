import os
import random
import gym
import torch
import numpy as np
import argparse
# Import pybullet environments (must use gym 0.21.0 for compatibility)
import pybullet_envs
from stable_baselines3.common.callbacks import EvalCallback
from ppo_rm import PPORM 
from networks import RewardModel

def train(args):
    log_path = "{}/rlhf_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(args.output_path, args.env_id, args.seed, args.lr, args.ppo_clip, 
                                                                                       args.rl_steps, args.hidden_layers, args.hidden_size, 
                                                                                       args.n_rm_epochs, args.rm_lr, args.rm_batch_size, 
                                                                                       args.enable_individualized_tau, args.quadratic_penalty, 
                                                                                       args.n_tau_iters, args.tau_init, args.tau_min, args.tau_max, args.rho)                                                         
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)

    env = gym.make(args.env_id)
    env.seed(args.seed)
    _ = env.reset()
    
    eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, n_eval_episodes=20, deterministic=True, render=False)
    
    reward_model = RewardModel(env, n_hidden_layers=args.hidden_layers, hidden_size=args.hidden_size)

    policy = PPORM("MlpPolicy", env, verbose=1, output_path=args.output_path, seed=args.seed, reward_model=reward_model, learning_rate=args.lr, clip_range=args.ppo_clip, 
                   n_rm_epochs=args.n_rm_epochs, rm_lr=args.rm_lr, rm_batch_size=args.rm_batch_size, enable_individualized_tau=args.enable_individualized_tau,
                   quadratic_penalty=args.quadratic_penalty, n_tau_iters=args.n_tau_iters, tau_init=args.tau_init, tau_min=args.tau_min, tau_max=args.tau_max, rho=args.rho,
                   use_attention_tau=args.use_attention_tau, attention_hidden_dim=args.attention_hidden_dim, attention_num_heads=args.attention_num_heads,
                   attention_num_layers=args.attention_num_layers, attention_lr=args.attention_lr, train_attention_freq=args.train_attention_freq)

    policy.learn(total_timesteps=args.rl_steps, callback=eval_callback, reset_num_timesteps=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="output path to save training results", default="/path/to/train/result/directory")
    parser.add_argument("--env_id", help="name of environment", default="AntBulletEnv-v0") 
    parser.add_argument("--seed", default=0, type=int)
    # for policy
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--ppo_clip", default=0.2, type=float)
    parser.add_argument("--rl_steps", default=3000000, type=int)
    # for reward model
    parser.add_argument("--hidden_layers", default=2, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--n_rm_epochs", default=1, type=int)
    parser.add_argument("--rm_lr", default=5e-4, type=float)
    parser.add_argument("--rm_batch_size", default=64, type=int)
    # for adaptive preference scaling
    parser.add_argument("--enable_individualized_tau", help="whether to use adaptive preference scaling", action='store_true')
    parser.add_argument("--quadratic_penalty", help="whether to apply quadratic regularization", action='store_true')
    parser.add_argument("--n_tau_iters", help="number of Newton iterations", default=0, type=int)
    parser.add_argument("--tau_init", help="initial value of the adaptive preference scaler", default=1.0, type=float)
    parser.add_argument("--tau_min", help="lower bound for the adaptive preference scaler", default=0.0, type=float)
    parser.add_argument("--tau_max", help="upper bound for the adaptive preference scaler", default=0.0, type=float)
    parser.add_argument("--rho", help="regularization parameter (rho0 in the paper)", default=0.0, type=float)
    # NEW: for attention-based tau network
    parser.add_argument("--use_attention_tau", help="use attention network for tau prediction", action='store_true')
    parser.add_argument("--attention_hidden_dim", help="hidden dimension for attention network", default=128, type=int)
    parser.add_argument("--attention_num_heads", help="number of attention heads", default=4, type=int)
    parser.add_argument("--attention_num_layers", help="number of transformer layers", default=2, type=int)
    parser.add_argument("--attention_lr", help="learning rate for attention network", default=1e-4, type=float)
    parser.add_argument("--train_attention_freq", help="train attention network every N epochs", default=5, type=int)
    args = parser.parse_args()
    train(args)