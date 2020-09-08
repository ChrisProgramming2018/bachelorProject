# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
 
import os
import sys 
import cv2
import time
import torch 
import random
import numpy as np
import robosuite as suite
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from agent import TQC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def create_next_obs(next_obs, size, args, state_buffer, policy, debug=False):
    state = next_obs["image"]
    state = state[:,:,0]
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    return obs, state_buffer 


def stacked_frames(state, size, args, policy, debug=False):
    state = state["image"]
    if debug:
        img = Image.fromarray(state, 'RGB')
        img.save('my.png')
        img.show()
        img = Image.fromarray(lum_img, 'L')
        img.save('my_gray.png')
    state = state[:,:,0]
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    zeros = torch.zeros_like(state)
    state_buffer = deque([], maxlen=args.history_length)
    for idx in range(args.history_length - 1):
        state_buffer.append(zeros)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    return obs, state_buffer

def get_state(state, size, args, perception, debug=False):
    state = state["image"]
    if debug:
        img = Image.fromarray(state, 'RGB')
        img.save('my.png')
        img.show()
        img = Image.fromarray(lum_img, 'L')
        img.save('my_gray.png')
    state = state[:,:,0]
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    zeros = torch.zeros_like(state)
    state_buffer = deque([], maxlen=args.history_length)
    for idx in range(args.history_length - 1):
        state_buffer.append(zeros)
    state = torch.stack(list(state_buffer), 0)
    obs = np.array(state)
    return obs, state_buffer

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


def evaluate_policy(policy, writer, total_timesteps, args, env, episode = 10):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    size = args.size
    use_gym = False
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if use_gym:
            env.seed(s)
            obs = env.reset()
            done = False
        else:
            obs, done = env.reset(), False
            obs, state_buffer = stacked_frames(obs, size, args, policy)

        for step in range(args.max_episode_steps):
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            if not use_gym:
                obs, state_buffer = create_next_obs(obs, size, args, state_buffer, policy)
            avg_reward += reward * args.reward_scalling
    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print ("---------------------------------------")
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
    return hours, mins, round(secs,2)



def train_agent(args):
    """

    Args:
    """
    
    # create CNN convert the [1,3,84,84] to [1, 200]
    
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pathname = str(args.locexp) + "/" + str(args.env_name) + '_agent_' + str(args.policy)
    pathname += "_batch_size_" + str(args.batch_size) + "_lr_act_" + str(args.lr_actor) 
    pathname += "_lr_critc_" + str(args.lr_critic) + "_lr_decoder_"
    pathname += '_n_quantiles_' + str(args.n_quantiles) + "_num_n_nets" + str(args.n_nets) + "_seed_" + str(args.seed)
    pathname += "_top_quantiles_to_drop_" + str(args.top_quantiles_to_drop_per_net)
    arg_text = str(args)
    write_into_file(pathname, arg_text) 
    tensorboard_name = str(args.locexp) + '/runs/' + pathname 
    writer = SummaryWriter(tensorboard_name)
    size = args.size
    env = suite.make(
            args.env_name,
            has_renderer=False,
            use_camera_obs=True,
            ignore_done=True,
            has_offscreen_renderer=True,
            camera_height=size,
            camera_width=size,
            render_collision_mesh=False,
            render_visual_mesh=True,
            camera_name='agentview',
            use_object_obs=False,
            camera_depth=True,
            reward_shaping=True,
            )
        
        

    state_dim = 200
    print("State dim, " , state_dim)
    action_dim = env.dof
    print("action_dim ", action_dim)
    max_action = 1
    args.target_entropy=-np.prod(action_dim)
    args.max_episode_steps = 200
    policy = TQC(state_dim, action_dim, max_action, args)    
    file_name = str(args.locexp) + "/pytorch_models/{}".format(args.env_name)
    obs_shape = (args.history_length, size, size)
    action_shape = (action_dim,)
    print("obs", obs_shape)
    print("act", action_shape)
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(args.buffer_size), args.image_pad, args.device)
    save_env_vid = False
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100) 
    episode_reward = 0
    evaluations = []
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
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num) 
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                
                print(text)
                write_into_file(pathname, text)
                #policy.train(replay_buffer, writer, episode_timesteps)
            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq 
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, args,  env))
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                save_model = file_name + '-{}reward_{:.2f}-agent{}'.format(episode_num, evaluations[-1], args.policy)
                policy.save(save_model)
            # When the training step is done, we reset the state of the environment
            state = env.reset()
            obs, state_buffer = stacked_frames(state, size, args, policy)

            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < args.start_timesteps:
            action = np.random.randn(env.dof)
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(obs)
 
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        done = float(done)
        new_obs, state_buffer = create_next_obs(new_obs, size, args, state_buffer, policy)
        
        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        done_bool = 0 if episode_timesteps + 1 == args.max_episode_steps else float(done)
        if episode_timesteps + 1 == args.max_episode_steps:
            done = True
        # We increase the total reward
        reward = reward * args.reward_scalling
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        if args.debug:
            print("add to buffer next_obs ", obs.shape)
            print("add to bufferobs ", new_obs.shape)
        replay_buffer.add(obs, action, reward, new_obs, done, done_bool)
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        if total_timesteps > args.start_timesteps:
            policy.train(replay_buffer, writer, 1)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, episode_num))
