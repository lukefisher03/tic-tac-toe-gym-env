import gym
from gym import spaces

action_space = spaces.MultiDiscrete([3,3])
print(action_space.sample())