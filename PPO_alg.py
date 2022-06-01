from cgi import test
import random
import gym
import Player2
from tic_tac_toe_gym.custom_env.tic_tac_toe import TicTacToe
from stable_baselines3 import PPO

p2 = Player2.Player2()
learning_env = gym.make('TicTacToe-v0',player2=p2)

TSTEPS = int(input("TSTEPS: "))

model = PPO('MlpPolicy', learning_env,learning_rate=0.0003,gamma=0.995, verbose=True)
model.learn(total_timesteps=TSTEPS)
model.save(f'models/PPO-{TSTEPS/1000}K-TSTEPS')

epis = 20
testing_env = gym.make('TicTacToe-v0', player2=p2)
obs = testing_env.reset()
for epi in range(epis):
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = testing_env.step(action)
        testing_env.render()
        
        if done:
            print('Done!')
    obs = testing_env.reset()

    print(f'Game #{epi} over\n\n\n')

print(f'Wins: {testing_env.wins}')
print(f'Losses: {testing_env.losses}')
print(f'Ties: {testing_env.ties}')
print(f'Cheats: {testing_env.invalidMoves}')
testing_env.close()