from typing import List, Tuple, Dict, Union, Optional, Sequence, Iterable
import os
import re
import time
import pickle
from pprint import pprint
from collections import namedtuple

import numpy as np
import gym
import gym_minigrid
from gym_minigrid.minigrid import MiniGridEnv, Door, Wall, Key, Goal

import matplotlib.pyplot as plt
import imageio
import random


Action = namedtuple('Action', ['MF', 'TL', 'TR', 'PK', 'UD'])
act = Action(0, 1, 2, 3, 4)
act_dict = act._asdict()
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
    env_path_lst = sorted(
        [
            os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)
            if os.path.isfile(os.path.join(env_folder, env_file))
        ]
    )
    path_dic = {}
    for path in env_path_lst:
        frac = re.split('[.-]', path)
        name = frac[2] + '-' + frac[3]
        path_dic[name] = path
    if verbose:
        pprint(path_dic)
    return path_dic


########################################
########################################
def init_agent_status(env, info: dict):
    """
    Get Init Agent Status (position, direction, front cell)
    """
    init_agent_pos = info['init_agent_pos']
    init_agent_dir = info['init_agent_dir']
    init_front_pos = init_agent_pos + init_agent_dir
    # Front type
    init_front_type = env.grid.get(init_front_pos[0], init_front_pos[1])
    ic(init_agent_pos)
    ic(init_agent_dir)
    ic(init_front_pos)
    ic(init_front_type)
    return init_agent_pos, init_agent_dir, init_front_pos, init_front_type


def init_door_status(env: MiniGridEnv, info: dict):
    """ Get Init Door Status (open or locked) """
    init_door_pos = info['door_pos']
    env_door: Door = env.grid.get(init_door_pos[0], init_door_pos[1])
    is_locked: int
    if env_door.is_open:
        is_locked = 0
    elif env_door.is_locked:
        is_locked = 1
    else:  # door condition is unknown
        is_locked = -1
    ic(init_door_pos)
    ic(is_locked)
    return env_door, init_door_pos, is_locked


########################################
########################################

def step_cost(env, action: int):
    """
    stage cost
    :param env:
    :param action:
    :return cost of action
    """
    assert isinstance(action, int), "action should be integer"
    assert 0 <= action <= 4, "action should in [0, 4]"

    front_cell_type: Union[Door, Wall, Key, Goal, None]

    state_cost = {
        "None": 0,
        "Wall": np.inf,
        "Goal": 0
    }
    action_cost = {
        "MF": 1,
        "TL": 1,
        "TR": 1,
        "PK": 1,
        "UD": 1
    }
    # Get the cell in front of the agent
    front_cell_pos = env.front_pos  # agent_pos + agent_dir

    if action == act.MF:
        front_cell_type = env.grid.get(front_cell_pos[0], front_cell_pos[1])
        cost = action_cost["MF"]
        if isinstance(front_cell_type, Wall):
            cost += state_cost["Wall"]
        elif isinstance(front_cell_type, Goal):
            cost += state_cost["Goal"]
        else:
            cost += state_cost["None"]
    else:
        cost = 1
    ic(cost)
    return cost


def step(env, action: int, render=False, verbose=False):
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

    # Get the cell in front of the agent
    front_cell_pos = env.front_pos  # agent_pos + agent_dir
    front_cell_type = env.grid.get(front_cell_pos[0], front_cell_pos[1])
    if isinstance(front_cell_type, Wall):
        msg = 'Wall'
    elif isinstance(front_cell_type, Goal):
        msg = "Goal"
    elif isinstance(front_cell_type, Door):
        msg = "Door"
    elif isinstance(front_cell_type, Key):
        msg = "Key"
    else:
        msg = "None"
    print(f"Front Cell: {msg}")
    ic(done)
    if render:
        plot_env(env)
    return step_cost(env, action), done


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

