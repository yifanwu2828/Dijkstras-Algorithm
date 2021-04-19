import argparse
from typing import Union
from collections import namedtuple
from functools import lru_cache
import os
from pprint import pprint
import random

import numpy as np
import gym
from gym_minigrid import minigrid
from gym_minigrid.minigrid import MiniGridEnv, Door, Wall, Key, Goal

try:
    from icecream import ic
    from icecream import install

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import utils

Action = namedtuple('Action', ['MF', 'TL', 'TR', 'PK', 'UD'])
act = Action(0, 1, 2, 3, 4)

act_dict = act._asdict()
inv_dict = {v: k for k, v in act_dict.items()}
# {
#     MF: 0,  # Move Forward
#     TL: 1,  # Turn Left
#     TR: 2,  # Turn Right
#     PK: 3,  # Pickup Key
#     UD: 4,  # Unlock Door
# }
front_cell_type: Union[Door, Wall, Key, Goal, None]


def doorkey_5x5_normal(env: MiniGridEnv):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
    Feel Free to modify this function
    """
    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    # return optim_act_seq
    pass


def main():
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this function
    """
    '''
    def partA():
        env_folder = './envs'
        env_path = './envs/example-8x8.env'
        env, info = utils.load_env(env_path)  # load an environment
        seq = doorkey_problem(env)  # find the optimal action sequence
        utils.draw_gif_from_seq(seq, utils.load_env(env_path)[0])  # draw a GIF & save
    
    
    def partB():
        env_folder = './envs/random_envs'
        env, info, env_path = utils.load_random_env(env_folder)
    '''
    # world_grid = minigrid.Grid.encode(env.grid)[:, :, 0].T.astype(np.float32), 1 is colormap, 2 state
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Env folder", type=str, default='./envs')
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument("-t", "--test", action="store_true", default=True, help="Test mode")
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()

    ############################
    # Config
    TEST = args.test
    # VERBOSE = args.verbose
    VERBOSE = False
    seed = args.seed
    env_folder = args.folder

    if VERBOSE:
        print(np.__version__)
    if TEST:
        # set overflow warning to error instead
        np.seterr(all='raise')

    ############################
    # Seed python and numpy RNG
    random.seed(seed)
    np.random.seed(seed)

    ############################
    # Obtain env path
    env_dict = utils.fetch_env_dict(env_folder, verbose=VERBOSE)
    env, info = utils.load_env(env_dict["5x5-normal"])
    # env, info = utils.load_env(env_dict["6x6-direct"])
    # env, info = utils.load_env(env_dict["6x6-normal"])
    # env, info = utils.load_env(env_dict["6x6-shortcut"])
    # env, info = utils.load_env(env_dict["8x8-direct"])
    # env, info = utils.load_env(env_dict["8x8-normal"])
    # env, info = utils.load_env(env_dict["8x8-shortcut"])

    if VERBOSE:
        print('\n<=====Environment Info =====>')
        ic(env.mission)
        # agent initial position & direction,
        # key position, door position, goal position
        pprint(info)  # Map size
        print('<===========================>')
        # Visualize the environment
        utils.plot_env(env)
    utils.plot_env(env)
    ############################
    # dimension
    height, width = info['height'], info['width']
    # agent info
    init_agent_pos, init_agent_dir, init_front_pos, init_front_type = utils.init_agent_status(env, info)
    init_agent_pos = np.flip(init_agent_pos)
    init_agent_dir = np.flip(init_agent_dir)
    init_front_pos = np.flip(init_front_pos)

    # door info
    env_door, init_door_pos, is_locked = utils.init_door_status(env, info)
    init_door_pos = np.flip(init_door_pos)
    # key info
    key_pos = info['key_pos']
    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    # goal info
    goal_pos = info['goal_pos']
    ############################
    # Map of object type to integers
    OBJECT_TO_IDX = {
        'empty': 1,
        'wall': 2,
        'door': 4,
        'key': 5,
        'goal': 8,
        'agent': 10,
    }
    # init map
    world_grid = minigrid.Grid.encode(env.grid)[:, :, 0].T  # .astype(np.float64)
    # print(world_grid)



    # if dist_goal != -1:
    #     pass
    # if dist_key != -1:
    #     # if door close, shortest pass to key, then shortest pass from key to goal
    #     pass
    # else:
    #     print("No Path Found!!")
    # ic(dist_goal)
    # ic(dist_key)
