import numpy as np
import gym
from pprint import pprint

from utils import *

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def example_use_of_gym_env():
    """
    The Coordinate System:
        (0,0): Top Left Corner
        (x,y): x-th column and y-th row
    """

    print('<========== Example Usages ===========> ')
    env_path = './envs/example-8x8.env'
    # env, info = load_env(env_path) # load an environment

    env, info = load_env('./envs/doorkey-8x8-shortcut.env')
    print('<Environment Info>\n')
    pprint(info)  # Map size
    # agent initial position & direction,
    # key position, door position, goal position
    print('<================>\n')

    # Visualize the environment
    plot_env(env)

    # Get the agent position
    agent_pos = env.agent_pos

    # Get the agent direction
    agent_dir = env.dir_vec  # or env.agent_dir

    # Get the cell in front of the agent
    front_cell = env.front_pos  # == agent_pos + agent_dir

    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3)  # NoneType, Wall, Key, Goal

    # Get the door status
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    is_open = door.is_open
    is_locked = door.is_locked

    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None

    # Take actions
    cost, done = step(env, MF)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'Moving Forward Costs: {cost}')
    cost, done = step(env, TL)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'Turning Left Costs: {cost}')
    cost, done = step(env, TR)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'Turning Right Costs: {cost}')
    cost, done = step(env, PK)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'Picking Up Key Costs: {cost}')
    cost, done = step(env, UD)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'Unlocking Door Costs: {cost}')

    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")

    # The number of steps so far
    print(f'Step Count: {env.step_count}')
