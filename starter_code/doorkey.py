import argparse
import sys
import os
from typing import Union, List
from collections import namedtuple
import random
from pprint import pprint

import numpy as np
import gym
from gym_minigrid import minigrid
from gym_minigrid.minigrid import MiniGridEnv, Door, Wall, Key, Goal

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import utils

Action = namedtuple('Action', ['MF', 'TL', 'TR', 'PK', 'UD'])
act = Action(0, 1, 2, 3, 4)
# {
#     MF: 0,  # Move Forward
#     TL: 1,  # Turn Left
#     TR: 2,  # Turn Right
#     PK: 3,  # Pickup Key
#     UD: 4,  # Unlock Door
# }
act_dict = act._asdict()
inv_act_dict = {v: k for k, v in act_dict.items()}

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
    # Act Seq Start -> Goal
    act_seq_start2goal, dir_seq_start2goal, cost_seq_start2goal = utils.action_recon(path=path_recon_start2goal,
                                                                                     init_dir=init_agent_dir)
    act_name_start2goal = [inv_act_dict[i] for i in act_seq_start2goal]
    if not cost_seq_start2goal:
        cost_start2goal = np.inf
    else:
        cost_start2goal = sum(cost_seq_start2goal)

    ####################################
    ####################################
    # Start to Key
    path_recon_start2key = utils.find_shortest_path(
        s=init_agent_pos,
        e=key_pos,
        grid=binary_grid_carried,
        direction=init_agent_dir
    )
    ic(path_recon_start2key)
    # Act Seq Start -> Key
    act_seq_start2key, dir_seq_start2key, cost_seq_start2key = utils.action_recon(path=path_recon_start2key,
                                                                                  init_dir=init_agent_dir)
    # act steps into key's pos
    act_seq_start2key.insert(-1, act.PK)
    dir_seq_start2key.insert(-1, dir_seq_start2key[-1])
    cost_seq_start2key.insert(-1, 1)

    # Remove MF
    act_seq_start2key.pop(-1)
    dir_seq_start2key.pop(-1)
    cost_seq_start2key.pop(-1)

    # obtain pos and dir before MF to key pos
    last_pos = path_recon_start2key[-2]
    last_dir = dir_seq_start2key[-1]

    act_name_start2key = [inv_act_dict[i] for i in act_seq_start2key]
    cost_start2key = sum(cost_seq_start2key)

    # Key to Door
    dist_from_key, prev_key = utils.dijkstra(s=last_pos, grid=binary_grid_open, direction=last_dir[-1])
    path_recon_key2door = utils.find_shortest_path(
        s=last_pos,
        e=door_pos,
        grid=binary_grid_open,
        direction=last_dir[-1]
    )
    # Act Seq Key -> Door
    act_seq_key2door, dir_seq_key2door, cost_seq_key2door = utils.action_recon(path=path_recon_key2door,
                                                                               init_dir=dir_seq_start2key[-1][-1])
    act_seq_key2door.insert(-1, act.UD)
    dir_seq_key2door.insert(-1, dir_seq_key2door[-1])
    cost_seq_key2door.insert(-1, 1)
    act_name_key2door = [inv_act_dict[j] for j in act_seq_key2door]
    cost_key2door = sum(cost_seq_key2door)

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
    cost_door2goal = sum(cost_seq_door2goal)

    ####################################
    ####################################
    cost_start2key2door2goal = np.sum([cost_start2key, cost_key2door, cost_door2goal], dtype=np.float32)

    opt_act_seq: List = []
    opt_act_name: List = []

    # if direct path exist and cost is less than find key
    if cost_start2goal < cost_start2key2door2goal <= np.inf:
        # Direct to goal
        print("Direct to Goal is Better!")
        opt_act_seq = act_seq_start2goal
        opt_act_name = act_name_start2goal

    # Compare the cost between (start -> goal) and (start -> key -> door -> goal)
    # Or, Direct Path DNE, find key
    elif cost_start2key2door2goal < cost_start2goal <= np.inf:
        if cost_start2goal < np.inf:
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


def doorkey_random_partB(env: MiniGridEnv, info: dict, verbose=False):
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
    # ic(binary_grid)

    # binary_grid with door locked key carried
    binary_grid_carried = binary_grid.copy()
    binary_grid_carried[tuple(key_pos)] = 0
    binary_grid[tuple(door_pos1)] = door1_is_locked
    binary_grid[tuple(door_pos2)] = door2_is_locked
    # ic(binary_grid_carried)

    # binary_grid with door1 open key used
    binary_grid_one_open = binary_grid.copy()
    binary_grid_one_open[tuple(door_pos1)] = 0
    binary_grid_one_open[tuple(door_pos2)] = 1
    binary_grid_one_open[tuple(key_pos)] = 0
    # ic(binary_grid_one_open)

    # binary_grid with door2 open key used
    binary_grid_two_open = binary_grid.copy()
    binary_grid_two_open[tuple(door_pos1)] = 1
    binary_grid_two_open[tuple(door_pos2)] = 0
    binary_grid_two_open[tuple(key_pos)] = 0
    # ic(binary_grid_two_open)

    # binary_grid with both door open key used
    binary_grid_both_open = binary_grid.copy()
    binary_grid_both_open[tuple(door_pos1)] = 0
    binary_grid_both_open[tuple(door_pos2)] = 0
    binary_grid_both_open[tuple(key_pos)] = 0
    # ic(binary_grid_both_open)

    # Start to Goal
    dist_from_start, prev_start = utils.dijkstra(s=init_agent_pos, grid=binary_grid_carried, direction=init_agent_dir)
    # ic(dist_from_start)
    path_recon_start2goal = utils.find_shortest_path(
        s=init_agent_pos,
        e=goal_pos,
        grid=binary_grid,
        direction=init_agent_dir
    )
    # Act Seq Start -> Goal
    act_seq_start2goal, dir_seq_start2goal, cost_seq_start2goal = utils.action_recon(path=path_recon_start2goal,
                                                                                     init_dir=init_agent_dir)
    act_name_start2goal = [inv_act_dict[i] for i in act_seq_start2goal]

    if not cost_seq_start2goal:
        cost_start2goal = np.inf
    else:
        cost_start2goal = sum(cost_seq_start2goal)
    # cost_start2goal = dist_from_start[tuple(goal_pos)]
    # ic(cost_start2goal)
    #################################################################################################################
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

    act_seq_start2key.pop(-1)
    dir_seq_start2key.pop(-1)
    cost_seq_start2key.pop(-1)

    # obtain pos and dir before MF to key pos
    last_pos = path_recon_start2key[-2]
    last_dir = dir_seq_start2key[-1]

    start2key = dist_from_start[tuple(key_pos)]
    cost_start2key = sum(cost_seq_start2key)
    # ic(cost_start2key)
    #################################################################################################################
    # Key to Door 1
    dist_from_key, prev_key = utils.dijkstra(s=key_pos, grid=binary_grid_one_open, direction=dir_seq_start2key[-1][-1])
    path_recon_key2door1 = utils.find_shortest_path(
        s=last_pos,
        e=door_pos1,
        grid=binary_grid_one_open,
        direction=last_dir[-1]
    )
    # Act Seq Key -> Door1
    act_seq_key2door1, dir_seq_key2door1, cost_seq_key2door1 = utils.action_recon(path=path_recon_key2door1,
                                                                                  init_dir=dir_seq_start2key[-1][-1])

    if door1_is_locked:
        act_seq_key2door1.insert(-1, act.UD)
        dir_seq_key2door1.insert(-1, dir_seq_key2door1[-1])
        cost_seq_key2door1.insert(-1, 1)
    act_name_key2door1 = [inv_act_dict[j] for j in act_seq_key2door1]
    cost_key2door1 = sum(cost_seq_key2door1)
    # ic(cost_key2door1)

    # key to door2
    path_recon_key2door2 = utils.find_shortest_path(
        s=last_pos,
        e=door_pos2,
        grid=binary_grid_two_open,
        direction=last_dir[-1]
    )
    # Act Seq Key -> Door2
    act_seq_key2door2, dir_seq_key2door2, cost_seq_key2door2 = utils.action_recon(path=path_recon_key2door2,
                                                                                  init_dir=dir_seq_start2key[-1][-1])
    if door2_is_locked:
        act_seq_key2door2.insert(-1, act.UD)
        dir_seq_key2door2.insert(-1, dir_seq_key2door1[-1])
        cost_seq_key2door2.insert(-1, 1)
    act_name_key2door2 = [inv_act_dict[j] for j in act_seq_key2door2]
    cost_key2door2 = sum(cost_seq_key2door2)
    # ic(cost_key2door2)
    #################################################################################################################

    # Door1 to Goal
    dist_from_door1, prev_door1 = utils.dijkstra(s=door_pos1, grid=binary_grid_one_open,
                                                 direction=dir_seq_key2door1[-1][-1])
    path_recon_door12goal = utils.find_shortest_path(
        s=door_pos1,
        e=goal_pos,
        grid=binary_grid_one_open,
        direction=dir_seq_key2door1[-1][-1]
    )
    # Act Seq Door1 -> Goal
    act_seq_door12goal, dir_seq_door12goal, cost_seq_door12goal = utils.action_recon(path=path_recon_door12goal,
                                                                                     init_dir=dir_seq_key2door1[-1][-1])
    act_name_door12goal = [inv_act_dict[k] for k in act_seq_door12goal]
    cost_door12goal = sum(cost_seq_door12goal)
    # ic(cost_door12goal)

    # Door2 to Goal
    dist_from_door2, prev_door2 = utils.dijkstra(s=door_pos2, grid=binary_grid_two_open,
                                                 direction=dir_seq_key2door2[-1][-1])
    path_recon_door22goal = utils.find_shortest_path(
        s=door_pos2,
        e=goal_pos,
        grid=binary_grid_two_open,
        direction=dir_seq_key2door2[-1][-1]
    )
    # Act Seq Door1 -> Goal
    act_seq_door22goal, dir_seq_door22goal, cost_seq_door22goal = utils.action_recon(path=path_recon_door22goal,
                                                                                     init_dir=dir_seq_key2door2[-1][-1])
    act_name_door22goal = [inv_act_dict[k] for k in act_seq_door22goal]
    cost_door22goal = sum(cost_seq_door22goal)
    # ic(cost_door22goal)

    through_door1 = cost_start2key + cost_key2door1 + cost_door12goal
    through_door2 = cost_start2key + cost_key2door1 + cost_door22goal

    ic(cost_start2goal)
    ic(through_door1)
    ic(through_door2)

    opt_act_seq = []
    opt_act_name = []
    # Both Door Open
    if door1_is_locked == 0 and door2_is_locked == 0:
        # Go direct to Goal
        print("Both Door Open. Direct to Goal is Better!")
        opt_act_seq = act_seq_start2goal
        opt_act_name = act_name_start2goal

    # One Door(1) open
    elif door1_is_locked == 0 and door2_is_locked == 1:
        # Door 1 is open and Door 2 is locked
        # Either find key and to door 2 or direct to goal
        if through_door2 < cost_start2goal <= np.inf:
            print("Find Key and go through Door 2 is Better!")
            comb_lst = [act_seq_start2key, act_seq_key2door2, act_seq_door22goal]
            for acs in comb_lst:
                opt_act_seq.extend(acs)
            opt_act_name.extend([act_name_start2key, act_name_key2door2, act_name_door22goal])

        elif cost_start2goal < through_door2 <= np.inf:
            # Go direct to Goal
            print("Door 1 Open. Direct to Goal is Better!")
            opt_act_seq = act_seq_start2goal
            opt_act_name = act_name_start2goal


    # One Door(2) open
    elif door1_is_locked == 1 and door2_is_locked == 0:
        # Door 2 is open and Door 1 is locked
        # Either find key and to door 1 or direct to goal
        if through_door1 < cost_start2goal <= np.inf:
            comb_lst = [act_seq_start2key, act_seq_key2door1, act_seq_door12goal]
            for acs in comb_lst:
                opt_act_seq.extend(acs)
            opt_act_name.extend([act_name_start2key, act_name_key2door1, act_name_door12goal])

        elif cost_start2goal < through_door1 <= np.inf:
            # Go direct to Goal
            print("Door 2 Open. Direct to Goal is Better!")
            opt_act_seq = act_seq_start2goal
            opt_act_name = act_name_start2goal
    # Both Door locked
    else:
        # Find key first, decide to go through door 1 or door 2
        print("Both Door Locked. Find Key is the only way!")
        if through_door1 < through_door2:
            print("Through Door 1")
            comb_lst = [act_seq_start2key, act_seq_key2door1, act_seq_door12goal]
            for acs in comb_lst:
                opt_act_seq.extend(acs)
            opt_act_name.extend([act_name_start2key, act_name_key2door1, act_name_door12goal])
        else:
            print("Through Door 2")
            comb_lst = [act_seq_start2key, act_seq_key2door2, act_seq_door22goal]
            for acs in comb_lst:
                opt_act_seq.extend(acs)
            opt_act_name.extend([act_name_start2key, act_name_key2door2, act_name_door22goal])
    if verbose:
        print(f"opt_act_seq: {opt_act_seq}")
        print(f"opt_act_name: {opt_act_name}")
    return opt_act_seq, opt_act_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Env folder", type=str, default='./envs')
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=False)
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
    A = True
    B = True
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
    # random.seed(seed)
    # np.random.seed(seed)
    ############################
    # Part A
    if A:
        # Obtain env path
        env_dict, env_name = utils.fetch_env_dict(env_folder, verbose=VERBOSE)
        # env_5x5_normal, info_5x5_normal = utils.load_env(env_dict["5x5-normal"])      # pass
        # env_6x6_direct, info_6x6_direct = utils.load_env(env_dict["6x6-direct"])      # pass
        # env_6x6_normal, info_6x6_normal = utils.load_env(env_dict["6x6-normal"])      # pass
        # env_6x6_shortcut, info_6x6_shortcut = utils.load_env(env_dict["6x6-shortcut"])    # pass
        # env_8x8_direct, info_8x8_direct = utils.load_env(env_dict["8x8-direct"])      # pass
        # env_8x8_normal, info_8x8_normal = utils.load_env(env_dict["8x8-normal"])      # pass
        # env_8x8_shortcut, info_8x8_shortcut = utils.load_env(env_dict["8x8-shortcut"])    # pass
        env_lst = []
        for key, value in env_dict.items():
            env, info = utils.load_env(value)
            env_lst.append((env, info, key))
        for x in env_lst:
            env, info, key = x
            ic(key)
            if VERBOSE:
                print('\n<=====Environment Info =====>')
                print(env.mission)
                pprint(info)  # Map size
                print('<===========================>')
                # Visualize the environment
                if args.render:
                    utils.plot_env(env)
            print(f'<=========== {key} =============>')
            opt_act_seq, opt_act_name = doorkey_partA(env, info, verbose=False)
            # utils.draw_gif_from_seq(seq=opt_act_seq, env=env, path=f'./gif/doorkey_{key}_demo.gif')
            for ac in opt_act_seq:
                try:
                    utils.step(env, ac, render=args.render)
                except KeyboardInterrupt:
                    sys.exit(0)
            print('<===============================>\n')
    ############################
    ############################

    if B:
        # Part B
        random_env_folder = os.path.join(env_folder, "random_envs")
        for i in range(30):
            env, info, env_path = utils.load_random_env(random_env_folder)
            # env.seed = seed
            print(env_path)

            if VERBOSE:
                print('\n<=====Environment Info =====>')
                print(env.mission)
                pprint(info)  # Map size
                print('<===========================>')
                # Visualize the environment
                if args.render:
                    utils.plot_env(env)
            ############################
            ############################
            print(f'<========================>')
            opt_act_seq, opt_act_name = doorkey_random_partB(env, info, verbose=False)
            # utils.draw_gif_from_seq(seq=opt_act_seq, env=env, path=f'./gif/random/doorkey{i}.gif')
            for ac in opt_act_seq:
                try:
                    utils.step(env, ac, render=args.render)
                except KeyboardInterrupt:
                    sys.exit(0)
            print('<===============================>\n')
    print("Done!")
