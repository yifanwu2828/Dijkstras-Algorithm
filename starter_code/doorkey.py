import argparse
import os.path
from typing import Union, List
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


def doorkey_partA(env: MiniGridEnv, info: dict, verbose=False):
    """
    """
    # agent info
    init_agent_pos = np.flip(info["init_agent_pos"])
    init_agent_dir = np.flip(info["init_agent_dir"])

    # door info
    door_pos = np.flip(info["door_pos"])
    is_locked = int(not info['door_open'][0])

    # key info
    key_pos = np.flip(info['key_pos'])

    # Determine whether agent is carrying a key
    is_carrying = True if env.carrying is not None else False

    # goal info
    goal_pos = np.flip(info['goal_pos'])
    ############################

    # init map
    world_grid = minigrid.Grid.encode(env.grid)[:, :, 0].T

    # binary_grid with door status(open:0, locked:1) and key status(carried:0 exist:1)
    binary_grid = np.where(world_grid != OBJECT_TO_IDX['wall'], 0, 1).astype("uint8")
    binary_grid[door_pos[0], door_pos[1]] = is_locked
    binary_grid[key_pos[0], key_pos[1]] = int(not is_carrying)
    # ic(binary_grid)

    # binary_grid with door locked key carried
    binary_grid_carried = binary_grid.copy()
    binary_grid_carried[door_pos[0], door_pos[1]] = is_locked
    binary_grid_carried[key_pos[0], key_pos[1]] = 0
    # ic(binary_grid_carried)

    # binary_grid with door open key used
    binary_grid_open = binary_grid.copy()
    binary_grid_open[door_pos[0], door_pos[1]] = 0
    binary_grid_open[key_pos[0], key_pos[1]] = 0
    # ic(binary_grid_open)

    ####################################
    ####################################

    # Start to Goal
    dist_from_start, prev_start = utils.dijkstra(s=init_agent_pos, grid=binary_grid_carried, direction=init_agent_dir)

    path_recon_start2goal = utils.find_shortest_path(
        s=init_agent_pos,
        e=goal_pos,
        grid=binary_grid,
        direction=init_agent_dir
    )
    # Act Seq Start -> Key
    act_seq_start2goal, dir_seq_start2goal, cost_seq_start2goal = utils.action_recon(path=path_recon_start2goal,
                                                                                     init_dir=init_agent_dir)
    act_name_start2goal = [inv_act_dict[i] for i in act_seq_start2goal]
    if verbose:
        assert len(act_seq_start2goal) == len(act_name_start2goal)
        for p, name in enumerate(act_name_start2goal):
            assert act_dict[name] == act_seq_start2goal[p]
    # ic(path_recon_start2goal)
    # ic(act_seq_start2goal)
    # ic(act_name_start2goal)
    # ic(dir_seq_start2goal)
    # ic(cost_seq_start2goal)

    ####################################
    ####################################
    # Start to Key
    path_recon_start2key = utils.find_shortest_path(
        s=init_agent_pos,
        e=key_pos,
        grid=binary_grid_carried,
        direction=init_agent_dir
    )

    # Act Seq Start -> Key
    act_seq_start2key, dir_seq_start2key, cost_seq_start2key = utils.action_recon(path=path_recon_start2key,

                                                                                  init_dir=init_agent_dir)
    act_seq_start2key.insert(-1, act.PK)
    dir_seq_start2key.insert(-1, dir_seq_start2key[-1])
    cost_seq_start2key.insert(-1, 1)
    act_name_start2key = [inv_act_dict[i] for i in act_seq_start2key]
    if verbose:
        assert len(act_seq_start2key) == len(act_name_start2key)
        for p, name in enumerate(act_name_start2key):
            assert act_dict[name] == act_seq_start2key[p]
    # ic(path_recon_start2key)
    # ic(act_seq_start2key)
    # ic(act_name_start2key)
    # ic(dir_seq_start2key)
    # ic(cost_seq_start2key)

    # Key to Door
    dist_from_key, prev_key = utils.dijkstra(s=key_pos, grid=binary_grid_open, direction=dir_seq_start2key[-1][-1])

    path_recon_key2door = utils.find_shortest_path(
        s=key_pos,
        e=door_pos,
        grid=binary_grid_open,
        direction=dir_seq_start2key[-1][-1]
    )
    # Act Seq Key -> Door
    act_seq_key2door, dir_seq_key2door, cost_seq_key2door = utils.action_recon(path=path_recon_key2door,
                                                                               init_dir=dir_seq_start2key[-1][-1])
    act_seq_key2door.insert(-1, act.UD)
    dir_seq_key2door.insert(-1, dir_seq_key2door[-1])
    cost_seq_key2door.insert(-1, 1)
    act_name_key2door = [inv_act_dict[j] for j in act_seq_key2door]
    if verbose:
        assert len(act_seq_key2door) == len(act_name_key2door)
        for p, name in enumerate(act_name_key2door):
            assert act_dict[name] == act_seq_key2door[p]

    # ic(dist_from_key)
    # ic(path_recon_key2door)
    # ic(act_seq_key2door)
    # ic(act_name_key2door)
    # ic(dir_seq_key2door)
    # ic(cost_seq_key2door)

    ####################################
    ####################################

    # Door to Goal
    dist_from_door, prev_door = utils.dijkstra(s=door_pos, grid=binary_grid_open, direction=dir_seq_key2door[-1][-1])
    path_recon_door2goal = utils.find_shortest_path(
        s=door_pos,
        e=goal_pos,
        grid=binary_grid_open,
        direction=dir_seq_key2door[-1][-1]
    )
    # Act Seq Door -> Goal
    act_seq_door2goal, dir_seq_door2goal, cost_seq_door2goal = utils.action_recon(path=path_recon_door2goal,
                                                                                  init_dir=dir_seq_key2door[-1][-1])
    act_name_door2goal = [inv_act_dict[k] for k in act_seq_door2goal]
    if verbose:
        assert len(act_seq_door2goal) == len(act_name_door2goal)
        for p, name in enumerate(act_name_door2goal):
            assert act_dict[name] == act_seq_door2goal[p]

    # ic(dist_from_door)
    # ic(path_recon_door2goal)
    # ic(act_seq_door2goal)
    # ic(act_name_door2goal)
    # ic(dir_seq_door2goal)
    # ic(cost_seq_door2goal)

    ####################################
    ####################################
    # ic(dist_from_start)
    # ic(dist_from_key)
    # ic(dist_from_door)

    start2goal = dist_from_start[tuple(goal_pos)]
    start2key = dist_from_start[tuple(key_pos)]
    key2door = dist_from_key[tuple(door_pos)]
    door2goal = dist_from_door[tuple(goal_pos)]

    start2key2door2goal = np.sum([start2key, key2door, door2goal], dtype=np.float32)
    # ic(start2goal)
    # ic(start2key)
    # ic(key2door)
    # ic(door2goal)
    # ic(start2key2door2goal)

    opt_act_seq: List = []
    opt_act_name: List = []

    # if direct path exist and cost is less than find key
    if start2goal < start2key2door2goal <= np.inf:
        # Direct to goal
        print("Direct to Goal is Bette!")
        opt_act_seq = act_seq_start2goal
        opt_act_name = act_name_start2goal

    # Compare the cost between (start -> goal) and (start -> key -> door -> goal)
    # Or, Direct Path DNE, find key
    elif start2key2door2goal < start2goal <= np.inf:
        if start2goal < np.inf:
            print("Find Key is Better!")
        else:
            print("Find Key is the only way!")
        comb_lst = [act_seq_start2key, act_seq_key2door, act_seq_door2goal]
        for acs in comb_lst:
            opt_act_seq.extend(acs)
        opt_act_name.extend([act_name_start2key, act_name_key2door, act_name_door2goal])
    else:
        print("No PATH FOUND!")
    if verbose:
        print(f"opt_act_seq: {opt_act_seq}")
        print(f"opt_act_name: {opt_act_name}")
    return opt_act_seq, opt_act_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Env folder", type=str, default='./envs')
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument("-t", "--test", action="store_true", default=True, help="Test mode")
    parser.add_argument("--logdir", type=str, default="./gif", help="Log directory")
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()

    ############################
    # Config
    TEST = args.test
    VERBOSE = args.verbose
    # VERBOSE = False
    seed = args.seed
    env_folder = args.folder
    logdir = args.logdir
    ############################
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
    # env_dict, env_name = utils.fetch_env_dict(env_folder, verbose=VERBOSE)
    # env, info = utils.load_env(env_dict["5x5-normal"])      # pass
    # env, info = utils.load_env(env_dict["6x6-direct"])      # pass
    # env, info = utils.load_env(env_dict["6x6-normal"])      # pass
    # env, info = utils.load_env(env_dict["6x6-shortcut"])    # pass
    # env, info = utils.load_env(env_dict["8x8-direct"])      # pass
    # env, info = utils.load_env(env_dict["8x8-normal"])      # pass
    # env, info = utils.load_env(env_dict["8x8-shortcut"])    # pass


    random_env_folder = os.path.join(env_folder, "random_envs")

    env, info, env_path = utils.load_random_env(random_env_folder)
    env.seed = seed
    print(env_path)


    if VERBOSE:
        print('\n<=====Environment Info =====>')
        print(env.mission)
        pprint(info)  # Map size
        print('<===========================>')
        # Visualize the environment
        utils.plot_env(env)
    ############################
    ############################
    # opt_act_seq, opt_act_name = doorkey_partA(env, info, verbose=True)

    # ep_cost = 0
    # done = False
    # for t, ac in enumerate(opt_act_seq):
    #     cost, done = utils.step(env, action=ac, render=True, verbose=True)
    #     ep_cost += cost
    #     # Determine whether we stepped into the goal
    #     if done:
    #         print("\nReached Goal\n")
    #
    # # The number of steps so far
    # print(f'Step Count: {env.step_count}')
    # print(f"Episode Cost: {ep_cost}")

    # agent info
    init_agent_pos = np.flip(info["init_agent_pos"])
    init_agent_dir = np.flip(info["init_agent_dir"])

    # door info
    door_pos1 = np.flip(info["door_pos"][0])
    door_pos2 = np.flip(info["door_pos"][1])
    door1_is_locked = int(not info["door_open"][0])
    door2_is_locked = int(not info["door_open"][1])

    # key info
    key_pos = np.flip(info['key_pos'])

    # Determine whether agent is carrying a key
    is_carrying = True if env.carrying is not None else False

    # goal info
    goal_pos = np.flip(info['goal_pos'])
    ############################

    # init map
    world_grid = minigrid.Grid.encode(env.grid)[:, :, 0].T

    # binary_grid with door status(open:0, locked:1) and key status(carried:0 exist:1)
    binary_grid = np.where(world_grid != OBJECT_TO_IDX['wall'], 0, 1).astype("uint8")
    binary_grid[tuple(door_pos1)] = door1_is_locked
    binary_grid[tuple(door_pos2)] = door2_is_locked
    binary_grid[tuple(key_pos)] = 1
    ic(binary_grid)

    # binary_grid with door locked key carried
    binary_grid_carried = binary_grid.copy()
    binary_grid_carried[tuple(key_pos)] = 0
    binary_grid[tuple(door_pos1)] = door1_is_locked
    binary_grid[tuple(door_pos2)] = door2_is_locked
    ic(binary_grid_carried)

    # binary_grid with door1 open key used
    binary_grid_one_open = binary_grid.copy()
    binary_grid_one_open[tuple(door_pos1)] = 0
    binary_grid_one_open[tuple(door_pos2)] = 1
    binary_grid_one_open[tuple(key_pos)] = 0
    ic(binary_grid_one_open)


    # binary_grid with door2 open key used
    binary_grid_two_open = binary_grid.copy()
    binary_grid_two_open[tuple(door_pos1)] = 1
    binary_grid_two_open[tuple(door_pos2)] = 0
    binary_grid_two_open[tuple(key_pos)] = 0
    ic(binary_grid_two_open)


    # binary_grid with both door open key used
    binary_grid_both_open = binary_grid.copy()
    binary_grid_both_open[tuple(door_pos1)] = 0
    binary_grid_both_open[tuple(door_pos2)] = 0
    binary_grid_both_open[tuple(key_pos)] = 0
    ic(binary_grid_both_open)

    # Both Door Open
    if door1_is_locked == 0 and door2_is_locked == 0:
        # Go direct to Goal
        pass
    # One Door(1) open
    elif door1_is_locked == 0 and door2_is_locked == 1:
        # Door 1 is open and Door 2 is locked
        # Either find key and to door 2 or direct to goal
        pass
    # One Door(2) open
    elif door1_is_locked == 1 and door2_is_locked == 0:
        # Door 2 is open and Door 1 is locked
        # Either find key and to door 1 or direct to goal
        pass
    # Both Door locked
    else:
        # Find key first, decide to go through door 1 or door 2
        pass