# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>

import time
import random
import gym

from datetime import datetime
from gym import wrappers
import numpy as np
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torch
from agent_average_v1 import TD31v1
from agent import TD3
from memory import ReplayBuffer



def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def evaluate_policy(policy, writer, total_timesteps, args, episode=10):
    """


    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    avg_reward = 0.
    seeds = [x for x in range(10)]
    for s in seeds:
        env.seed(s)
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward




def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def time_format(sec):
    """

    Args:
        param1():

    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, secs



def train(args):
    """

    Args:
        param1(args): hyperparameter
    """

    # in case seed experements
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pathname = str(args.env_name) + '_repeat_train_' + str(args.repeat_opt)
    pathname += '_update_freq_' + str(args.target_update_freq) + "_num_q_target_"
    pathname += str(args.num_q_target) + "_seed_" + str(args.seed) + "_run_" + str(args.run) + "_agent_" + args.agent
    text = "Star_training target_update_freq: {}  num_q_target: {}  use device {} ".format(args.target_update_freq, args.num_q_target, args.device)
    print(pathname, text)
    write_into_file(pathname, text)
    arg_text = str(args)
    write_into_file(pathname, arg_text)
    tensorboard_name = 'runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(state_dim)
    if args.agent == "TD3_ad":
        policy = TD31v1(state_dim, action_dim, max_action, args)
    elif args.agent == "TD3":
        policy = TD3(state_dim, action_dim, max_action, args)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100)
    episode_reward = 0
    evaluations = []
    file_name = "%s_%s_%s" % ("TD3", args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")
    # We start the main loop over 500,000 timesteps
    tb_update_counter = 0
    while total_timesteps <  args.max_timesteps:
        tb_update_counter += 1
        # If the episode is done
        if done:
            episode_num += 1
            #env.seed(random.randint(0, 100))
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > args.tensorboard_freq:
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                text = "Total Timesteps: {} Episode Num: {} Reward: {}  Average Re: {:.2f} Time: {}".format(total_timesteps, episode_num, episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                print(text)
                write_into_file('search-' + pathname, text)
                for i in range(args.repeat_opt):
                    policy.train(replay_buffer, writer, episode_timesteps)
            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= args.eval_freq:
                policy.save("%s" % (file_name), directory="./pytorch_models")
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, episode_num))
                save_model = file_name + '-{}'.format(episode_num)
                policy.save(save_model, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)
            # When the training step is done, we reset the state of the environment
            obs = env.reset()
            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)


        if args.agent == "TD3_ad":
            if total_timesteps % args.target_update_freq == 0:
                policy.hardupdate()
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == 1000 else float(done)
        # We increase the total reward
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, episode_num))
    if args.save_model: 
        policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)

