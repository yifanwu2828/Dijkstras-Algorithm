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
# Map of object type to integers
OBJECT_TO_IDX = {
    'empty': 1,
    'wall': 2,
    'door': 4,
    'key': 5,
    'goal': 8,
    'agent': 10,
}
front_cell_type: Union[Door, Wall, Key, Goal, None]


def doorkey(env: MiniGridEnv, info: dict):
    """
    """
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
        print(env.mission)
        pprint(info)  # Map size
        print('<===========================>')
        # Visualize the environment
        utils.plot_env(env)
    utils.plot_env(env)
    ############################
    ############################
    # doorkey(env, info)
    # agent info
    init_agent_pos, init_agent_dir, init_front_pos, init_front_type = utils.init_agent_status(env, info)
    init_agent_pos = np.flip(init_agent_pos)
    init_agent_dir = np.flip(init_agent_dir)
    # door info
    env_door, init_door_pos, is_locked = utils.init_door_status(env, info)
    init_door_pos = np.flip(init_door_pos)

    # key info
    key_pos = info['key_pos']
    key_pos = np.flip(key_pos)

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None

    # goal info
    goal_pos = info['goal_pos']
    goal_pos = np.flip(goal_pos)
    ############################

    # init map
    world_grid = minigrid.Grid.encode(env.grid)[:, :, 0].T

    binary_grid = np.where(world_grid != OBJECT_TO_IDX['wall'], 0, 1).astype("uint8")
    binary_grid[init_door_pos[0], init_door_pos[1]] = is_locked

    dist_from_start, prev = utils.dijkstra(s=init_agent_pos, grid=binary_grid, direction=init_agent_dir)
    ic(dist_from_start)
    ic(prev)

    # Check if direct path exist
    dist2goal = dist_from_start[tuple(goal_pos)]
    dist2key = dist_from_start[tuple(key_pos)]

    binary_grid_open = binary_grid.copy()
    binary_grid_open[init_door_pos[0], init_door_pos[1]] = 0
    binary_grid_open[key_pos[0], key_pos[1]] = 0
    dist_from_key, _ = utils.dijkstra(s=key_pos, grid=binary_grid_open, direction=init_agent_dir)

    # TODO: get key act seq and open door seq and their cost
    key2door = dist_from_key[tuple(init_door_pos)]
    dist_from_door, _ = utils.dijkstra(s=init_door_pos, grid=binary_grid_open, direction=init_agent_dir)
    door2goal = dist_from_door[tuple(goal_pos)]
    ic(dist2goal)
    ic(dist2key)
    ic(key2door)
    ic(door2goal)
    start2key2goal = dist2key + key2door + door2goal
    ic(start2key2goal)

    #     # if door close, shortest pass to key, then shortest pass from key to goal
    #     pass
    # else:
    #     print("No Path Found!!")

    path_recon = utils.find_shortest_path(s=init_agent_pos, e=goal_pos, grid=binary_grid, direction=init_agent_dir)
    ic(path_recon)

    act_seq = utils.action_recon(path=path_recon, init_dir=init_agent_dir)
    act_name_seq = [inv_act_dict[i] for i in act_seq]
    ic(act_seq)
    ic(act_name_seq)

    # for ac in act_seq:
    #     utils.step(env, action=ac, render=True, verbose=True)