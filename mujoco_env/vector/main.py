# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>

import os
import sys
import time
import random
import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
from utilis import train
import multiprocessing as mp
from functools import partial


def main(arg):
    """ Starts different tests

    Args:
        param1(args): args

    """
    print(arg)
    path = arg.locexp
    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    dir_model = os.path.join(path, "pytorch_models")
    if arg.save_model and not os.path.exists(dir_model):
        os.makedirs(dir_model)
    print("Created model dir {} ".format(dir_model))
    processes = []
    pool = mp.Pool()
    seeds = [1,2,3,4]
    func = partial(train, arg)
    pool.map(func, seeds)
    pool.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="HalfCheetah-v3", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--start_timesteps', default=1e4, type=int)     # amount of steps before update the weights
    parser.add_argument('--eval_freq', default=10000, type=int)         # How often the evaluation step is performed (after how many timesteps)
    parser.add_argument('--max_timesteps', default=1e5, type=int)       # Total number of iterations/timesteps
    parser.add_argument('--save_model', default=True, type=bool)        # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--lr_critic', default=1e-4, type=float)        # learning rate critic
    parser.add_argument('--lr_actor', default=1e-4, type=float)         # learning rate actor
    parser.add_argument('--expl_noise', default=0.1, type=float)        # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--batch_size', default=100, type=int)          # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)         # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type=float)             # Target network update rate
    parser.add_argument('--policy_noise', default=0.2, type=float)      # STD of Gaussian noise added to the actions for the exploration purposes
    parser.add_argument('--noise_clip', default=0.5, type=float)        # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--policy_freq', default=2, type=int)           # Number of iterations to wait before the policy network (Actor model) is updated
    parser.add_argument('--target_update_freq', default=100, type=int)  # steps between the hard target upate for the critics
    parser.add_argument('--num_q_target', default=6, type=int)          # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=1000, type=int)   # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)           # amount of qtarget nets
    parser.add_argument('--reward_scalling', default=10, type=int)      # multiply the given reward
    parser.add_argument('--max_episode_steps', default=200, type=int)   # max steps per epiosde
    parser.add_argument('--locexp', type=str)                           # name of dir to save data
    parser.add_argument('--agent', default="TD3", type=str)             # select agent
    arg = parser.parse_args()
    main(arg)
