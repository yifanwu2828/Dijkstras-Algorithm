import argparse
from typing import Union
from collections import namedtuple

import random

import numpy as np
import gym
from gym_minigrid import minigrid
from gym_minigrid.minigrid import MiniGridEnv, Door, Wall, Key, Goal
from icecream import ic
from pprint import pprint

import utils

Action = namedtuple('Action', ['MF', 'TL', 'TR', 'PK', 'UD'])
act = Action(0, 1, 2, 3, 4)

act_dict = act._asdict()
inv_act_dict = {v: k for k, v in act_dict.items()}
# {
#     MF: 0,  # Move Forward
#     TL: 1,  # Turn Left
#     TR: 2,  # Turn Right
#     PK: 3,  # Pickup Key
#     UD: 4,  # Unlock Door
# }
front_cell_type: Union[Door, Wall, Key, Goal, None]






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
    # env, info = utils.load_env(env_dict["5x5-normal"])
    env, info = utils.load_env(env_dict["6x6-direct"])
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
    ############################
    # dimension
    height, width = info['height'], info['width']
    # agent info
    init_agent_pos, init_agent_dir, init_front_pos, init_front_type = utils.init_agent_status(env, info)
    init_agent_pos = np.flip(init_agent_pos)
    init_agent_dir = np.flip(init_agent_dir)
    init_front_pos = np.flip(init_front_pos)
    ic(init_agent_pos)
    ic(init_agent_dir)
    ic(init_front_pos)
    ic(init_front_type)

    # door info
    env_door, init_door_pos, is_locked = utils.init_door_status(env, info)
    init_door_pos = np.flip(init_door_pos)
    ic(init_door_pos)
    ic(is_locked)

    # key info
    key_pos = info['key_pos']
    key_pos = np.flip(key_pos)

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None

    # goal info
    goal_pos = info['goal_pos']
    goal_pos = np.flip(goal_pos)
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
    world_grid = minigrid.Grid.encode(env.grid)[:, :, 0].T
    # ic(world_grid)

    binary_grid = np.where(world_grid != OBJECT_TO_IDX['wall'], 0, 1).astype("uint8")
    binary_grid[init_door_pos[0], init_door_pos[1]] = is_locked
    binary_grid[init_door_pos[0], init_door_pos[1]] = is_locked

    distance, prev = utils.dijkstra(s=init_agent_pos, grid=binary_grid, direction=init_agent_dir)
    ic(distance)
    ic(prev)

    path_recon = utils.find_shortest_path(s=init_agent_pos, e=(4, 2), grid=binary_grid, direction=init_agent_dir)
    ic(path_recon)

    act_seq = utils.action_recon(path=path_recon, init_dir=init_agent_dir)
    # ic(act_seq)
    actname_seq = [inv_act_dict[i] for i in act_seq]
    ic(actname_seq)