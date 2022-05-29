from gym.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='tic_tac_toe_gym.custom_env.tic_tac_toe:TicTacToe',
    max_episode_steps=5
)

