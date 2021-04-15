import argparse
from typing import List, Tuple, Dict, Union, Optional
from collections import namedtuple, OrderedDict
from functools import lru_cache
import os
from pprint import pprint
import random

import numpy as np
import gym
from gym_minigrid.minigrid import MiniGridEnv, Door, Wall, Key, Goal

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
try:
    from icecream import install
    install()
except ImportError:
    pass

import utils
from example import example_use_of_gym_env

Action = namedtuple('Action', ['MF', 'TL', 'TR', 'PK', 'UD'])
act = Action(0, 1, 2, 3, 4)
act_dict = act._asdict()


# {
#     MF: 0,  # Move Forward
#     TL: 1,  # Turn Left
#     TR: 2,  # Turn Right
#     PK: 3,  # Pickup Key
#     UD: 4,  # Unlock Door
# }

def agent_status(env: MiniGridEnv):
    """ Get Agent Status (position, direction, front cell) """
    # Get the agent position
    agent_position = env.agent_pos
    # Get the agent direction
    agent_direction = env.dir_vec  # or env.agent_dir
    # Get the cell in front of the agent
    front_cell = env.front_pos  # agent_pos + agent_dir
    return agent_position, agent_direction, front_cell


def door_status(env: MiniGridEnv):
    """ Get Door status (open or locked) """
    # Get the door status
    env_door: Door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    return env_door.is_open, env_door.is_locked


def doorkey_problem(env: MiniGridEnv):
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
    # optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    # return optim_act_seq
    pass


def main():
    """"""
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

    TEST = args.test
    VERBOSE = args.verbose
    seed = args.seed
    env_folder = args.folder

    if VERBOSE:
        print(np.__version__)
    if TEST:
        # set overflow warning to error instead
        np.seterr(all='raise')

    # Seed python and numpy RNG
    random.seed(seed)
    np.random.seed(seed)

    # Obtain env path
    env_dict = utils.fetch_env_dict(env_folder, verbose=VERBOSE)
    env, info = utils.load_env(env_dict["5x5-normal"])

    if VERBOSE:
        print('\n<=====Environment Info =====>')
        ic(env.mission)
        # agent initial position & direction,
        # key position, door position, goal position
        pprint(info)  # Map size
        print('<===========================>')
        # Visualize the environment
        utils.plot_env(env)

    agent_pos, agent_dir, front_cell = agent_status(env)
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    is_open, is_locked = door_status(env)


    # Access the cell at coord: (2,3)
    cell: Union[Door, Wall, Key, Goal, None] = env.grid.get(3, 3)  # NoneType, Door, Wall, Key, Goal

    ic(cell)
    ic(agent_pos)
    ic(agent_dir)
    ic(front_cell)
    ic(door)
    ic(is_open)
    ic(is_locked)

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    ic(is_carrying)

    cost, done = utils.step(env, act.MF)
    # Determine whether we stepped into the goal
    if done:
        ic("Reached Goal")
    # The number of steps so far
    print(f'Step Count: {env.step_count}')