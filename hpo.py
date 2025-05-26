

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.mountain_env import CoppeliaMountainEnv
from env_runner import setup_environment

# --- Hyperparameter optimization functions ---
def objective(trial, env, n_steps=10_000):
    # Sample training hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 1.0)

    # --- Search over network architecture ---
    # Sample number of layers and hidden units per layer
    n_layers = trial.suggest_int('n_layers', 2, 4)  # Try 2 to 4 hidden layers
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])

    # Build architecture
    layer_sizes = [hidden_size] * n_layers
    policy_kwargs = dict(
        net_arch=dict(pi=layer_sizes, vf=layer_sizes)  
    )

    # Create the model
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        verbose=1,
        n_steps=500,
        policy_kwargs=policy_kwargs
    )

    # Train and evaluate
    model.learn(total_timesteps=n_steps)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    return -mean_reward  # Optuna minimizes


def run_hpo(env, n_trials=20, n_steps=10_000):
    '''
    Runs hyperparameter optimization for PPO on the given env.
    '''
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, env, n_steps), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (negative reward): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return study

# --- Environment creation helper ---
def make_env(mode='normal', use_camera=True, headless=False):
    '''
    Sets up the CoppeliaMountainEnv via env_runner's setup_environment.
    '''
    env = setup_environment(
        use_camera_setup=use_camera,
        launch_headless=headless,
        mode=mode
    )
    return env

# --- Example usage ---
if __name__ == '__main__':
    # Create environment
    env = make_env(mode='joints_only', use_camera=False, headless=False)
    # Run HPO: 20 trials, 10k steps each
    study = run_hpo(env, n_trials=20, n_steps=10_000)
    # Access best parameters
    print("Optimal hyperparameters found:", study.best_params)


"'learning_rate': 0.0007512939246111457, 'gamma': 0.9107641813721379, 'clip_range': 0.33032683745309116, 'n_epochs': 2, 'gae_lambda': 0.9839114450956908}"
"'learning_rate': 3.425664968432371e-05, 'gamma': 0.9156314216993051, 'clip_range': 0.35813302946093184, 'n_epochs': 7, 'gae_lambda': 0.8160372224072454"