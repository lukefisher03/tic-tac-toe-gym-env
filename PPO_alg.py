import gym
import Player2
from tic_tac_toe_gym.custom_env.tic_tac_toe import TicTacToe
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import A2C


p2 = Player2.Player2()
learning_env = gym.make('TicTacToe-v0',player2=p2)
TSTEPS = int(input("TSTEPS: "))

model = PPO('MlpPolicy', learning_env,learning_rate=0.0003,gamma=0.999, verbose=True, tensorboard_log='./tensor_board/')
model.learn(total_timesteps=TSTEPS)
model.save(f'models/PPO-OPTIMIZER')
