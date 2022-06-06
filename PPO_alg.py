from random import sample
import gym
import optuna
from Player2 import Player2
from tic_tac_toe_gym.custom_env.tic_tac_toe import TicTacToe
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

'''
Steps for implementing Optuna:

    1. Wrap model learning and instantiation in objective function. Params: trial -> optuna.Trial
    2. return EvalCallback.last_mean_reward in objective function.
    3. Specify hyperparameter suggestions for optuna. Pass into model.learn from sb3
    4. Create trial and initalize sampler and pruner
    5. Output study.best_trial.params to see optimized hyperparameters specified in step 3.

'''
    
def objective(trial: optuna.Trial):
    p2 = Player2(agent=0)
    learning_env = gym.make('TicTacToe-v0',player2=p2)
    # TSTEPS = int(input("TSTEPS: "))
    TSTEPS = 100000
    gamma = 1-trial.suggest_float('gamma', 0.00001, 0.1)
    trial.set_user_attr('gamma_', gamma)

    model = PPO('MlpPolicy', learning_env,learning_rate=0.0003,gamma=gamma, verbose=0)
    eval_callback = EvalCallback(learning_env)
    model.learn(total_timesteps=TSTEPS, callback=eval_callback)
    model.save(f'GAN/Agent1')
    return eval_callback.last_mean_reward

sampler = TPESampler(n_startup_trials=5)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2//3)
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
study.optimize(objective, n_trials=10, timeout=600)


trial = study.best_trial
print(f'Best Trial: {study.best_trial}')
print(trial.params)