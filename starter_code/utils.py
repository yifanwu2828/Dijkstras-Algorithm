import os
from typing import List, Tuple, Dict, Optional, Sequence, Iterable
import time
import re
from pprint import pprint

import numpy as np
import gym
import gym_minigrid
import pickle
import matplotlib.pyplot as plt
import imageio
import random


MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


############################################
############################################

def tic(message: Optional[str] = None) -> float:
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


############################################
############################################

def toc(t_start: float, name: Optional[str] = "Operation", ftime=False) -> None:
    """ Timing Function """
    assert isinstance(t_start, float)
    sec: float = time.time() - t_start
    if ftime:
        duration = time.strftime("%H:%M:%S", time.gmtime(sec))
        print(f'\n############ {name} took: {str(duration)} ############\n')
    else:
        print(f'\n############ {name} took: {sec:.4f} sec. ############\n')


########################################################################################

def fetch_env_dict(env_folder: str = './envs', verbose=False):
    """
    :param env_folder: folder contain .env files
    :param verbose: show msg
    return env_dict
    """
    assert isinstance(env_folder, str)
    assert os.path.isdir(env_folder)
    env_path_lst = [
        os.path.join(env_folder, env_file) for env_file in sorted(os.listdir(env_folder))
        if os.path.isfile(os.path.join(env_folder, env_file))
    ]
    path_dic = {}
    for path in env_path_lst:
        frac = re.split('[.-]', path)
        name = frac[2] + '-' + frac[3]
        path_dic[name] = path
    if verbose:
        pprint(path_dic)
    return path_dic


########################################################################################

def step_cost(action: int):
    """
    stage cost
    :param action:
    :return cost of action
    """
    # TODO:
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    return 0  # the cost of action


def step(env, action: int):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle
    }

    _, _, done, _ = env.step(actions[action])
    return step_cost(action), done


def generate_random_env(seed, task: str):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)

    # Seed python RNG
    random.seed(seed)

    # Seed numpy RNG
    np.random.seed(seed)

    env = gym.make(task)
    env.seed(seed)
    env.reset()
    return env


def load_env(path: str):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            env = pickle.load(f)
    else:
        raise ValueError("File not Found!!")

    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec
    }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])

            elif isinstance(env.grid.get(j, i), gym_minigrid.minigrid.Door):
                info['door_pos'] = np.array([j, i])

            elif isinstance(env.grid.get(j, i), gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])    
            
    return env, info


def load_random_env(env_folder: str):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    env_path = random.choice(env_list)
    if os.path.isfile(env_path):
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
    else:
        raise ValueError("File not Found!!")

    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec,
        'door_pos': [],
        'door_open': [],
    }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), gym_minigrid.minigrid.Door):
                info['door_pos'].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info['door_open'].append(True)
                else:
                    info['door_open'].append(False)
            elif isinstance(env.grid.get(j, i), gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])    
            
    return env, info, env_path


def save_env(env, path: str):
    """ Save env"""
    with open(path, 'wb') as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render('rgb_array', tile_size=32)
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq: Sequence[int], env, path: str = './gif/doorkey.gif'):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]
    
    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode='I', duration=0.8) as writer:
        img = env.render('rgb_array', tile_size=32)
        writer.append_data(img)
        for act in seq:
            img = env.render('rgb_array', tile_size=32)
            step(env, act)
            writer.append_data(img)
    print(f'GIF is written to {path}')
    return

