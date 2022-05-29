from gym.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='CustomEnv.custom_env:TicTacToe',
    max_episode_steps=5
)

