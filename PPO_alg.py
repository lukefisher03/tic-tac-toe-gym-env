import gym
import optuna
from Player2 import Player2
from tic_tac_toe_gym.custom_env.tic_tac_toe import TicTacToe
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float('gamma', 0.0001, 0.1, log=1)
    learning_rate = 1.0 - trial.suggest_float('learning_rate', 0.0003, 0.1, log=1)

    trial.set_user_attr('gamma_', gamma)

    return {
        'gamma': gamma,
        'learning_rate': learning_rate
    }


class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=deterministic,
                verbose=verbose
        )
        
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True



def objective(trial: optuna.Trial) -> float:

    p2 = Player2(agent=0)
    learning_env = gym.make('TicTacToe-v0',player2=p2)
    TSTEPS = int(input("TSTEPS: "))

    model = PPO('MlpPolicy', learning_env,learning_rate=0.0003,gamma=0.999, verbose=True)
    model.learn(total_timesteps=TSTEPS)
    model.save(f'GAN/Agent1')









