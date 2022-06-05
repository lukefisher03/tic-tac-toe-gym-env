import gym
from Player2 import Player2
from tic_tac_toe_gym.custom_env.tic_tac_toe import TicTacToe
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


p2 = Player2(agent=0)
learning_env = gym.make('TicTacToe-v0',player2=p2)
TSTEPS = int(input("TSTEPS: "))

model = PPO('MlpPolicy', learning_env,learning_rate=0.0003,gamma=0.999, verbose=True)
model.learn(total_timesteps=TSTEPS)
model.save(f'GAN/Agent1')
