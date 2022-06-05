import gym
from stable_baselines3 import PPO
from tic_tac_toe_gym.custom_env.tic_tac_toe import TicTacToe
from Player2 import Player2

p2 = Player2(agent=2)
env = gym.make('TicTacToe-v0', player2=p2)

model = PPO.load('./GAN/Agent1.zip')
epis = 10000

obs = env.reset()
for _ in range(epis):
    done = False

    while not done:
        action, _state = model.predict(obs)

        obs, reward, done, info = env.step(action)
        #env.render()

    obs = env.reset()

print(f'Wins: {env.wins}')
print(f'Losses: {env.losses}')
print(f'Ties: {env.ties}')
print(f'Invalid Moves: {env.invalidMoves}')
