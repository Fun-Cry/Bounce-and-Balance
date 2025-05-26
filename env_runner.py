import gymnasium as gym
import time
import os
import numpy as np # Added for reward logging
import matplotlib.pyplot as plt # Added for plotting
from env.mountain_env import CoppeliaMountainEnv
# from utils.coppelia_launcher import start_coppeliasim, stop_coppeliasim # Use the launcher
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback # Added BaseCallback

# --- Default Configurations ---
DEFAULT_SCENE_CAMERA = "rex_camera.ttt"
DEFAULT_SCENE_LIDAR = "rex_lidar.ttt" # Assuming you have this scene file
DEFAULT_ENV_MAX_STEPS = 500 # Max steps per episode for the Gym env

DEFAULT_MOUNTAIN_PARAMS_FOR_ENV = { # These are passed to CoppeliaMountainEnv's scene_params
    'n_items': 1, # Number of mountains/obstacles
    'mountain_peak_radius': 0.35,
    'mountain_base_radius_factor_range': (2.5, 4.5),
    # Other params will use CoppeliaMountainEnv's internal defaults if not specified
}

class RewardLoggerCallback(BaseCallback):
    """
    A custom callback that logs rewards and timesteps.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.timesteps = []
        self.current_episode_rewards = []

    def _on_step(self) -> bool:
        # Log reward at each step for the current episode
        # Assuming the info dict contains episode reward when done
        if 'episode' in self.locals['infos'][0]:
            episode_info = self.locals['infos'][0]['episode']
            if 'r' in episode_info: # 'r' is the cumulative reward for the episode
                self.rewards.append(episode_info['r'])
                self.timesteps.append(self.num_timesteps)
                if self.verbose > 0:
                    print(f"Logged episode reward: {episode_info['r']} at timestep {self.num_timesteps}")
        return True

def print_manual_intervention_instructions(scene_file_name):
    print("\n" + "="*60)
    print("MANUAL COPPELIASIM INTERVENTION MAY BE REQUIRED")
    print("="*60)
    print(f"Auto-launch might have failed or CoppeliaSim needs manual setup.")
    print(f"1. Ensure CoppeliaSim is running.")
    print(f"2. Open the scene: '{scene_file_name}'")
    print(f"3. Ensure ZMQ Remote API server is started in the scene (e.g., simRemoteApi.start(19997)).")
    print(f"4. Ensure simulation in CoppeliaSim is STOPPED.")
    input("\n>>> Press Enter here AFTER CoppeliaSim is manually prepared or if auto-launch was successful...\n")

def setup_environment(use_camera_setup=True, custom_env_config=None, launch_headless=False, mode='normal'): # Added mode
    """
    Sets up and returns the Gym environment.
    Handles one-time CoppeliaSim launch.
    'mode' can be 'normal' or 'joints_only'.
    """
    # --- CoppeliaSim Launch (Simplified: Assuming it's handled or user does it manually) ---
    # It's often better to start CoppeliaSim once outside the script for stability during development.
    # For this example, we'll assume CoppeliaSim is running and the correct scene is open.
    # You can uncomment and adapt the start_coppeliasim logic if needed.

    if use_camera_setup:
        print(f"CONFIG: Using CAMERA setup. Mode: {mode}")
        # selected_scene = DEFAULT_SCENE_CAMERA # Keep for reference if auto-launching
        sensor_cfg = {'type': 'camera', 'resolution': (64, 64)}
        render_cfg = "human" if not launch_headless else None
    else: # Lidar setup
        print(f"CONFIG: Using LIDAR setup. Mode: {mode}")
        # selected_scene = DEFAULT_SCENE_LIDAR # Keep for reference if auto-launching
        sensor_cfg = {'type': 'lidar', 'max_points': 500}
        render_cfg = None # No human render for lidar in this basic setup

    # --- Automatic CoppeliaSim Launch (Optional, can be tricky) ---
    # scene_to_launch = selected_scene
    # if not start_coppeliasim(scene_to_launch, headless=launch_headless):
    #     print_manual_intervention_instructions(scene_to_launch)
    # else:
    #     print(f"Proceeding to connect to CoppeliaSim (assumed scene: '{scene_to_launch}').")
    #     time.sleep(3) # Give ZMQ time to initialize if sim was just launched

    print(f"Attempting to connect to CoppeliaSim. Please ensure it's running with the appropriate scene and ZMQ server.")


    # Prepare parameters for CoppeliaMountainEnv initialization
    env_init_params = {
        "render_mode": render_cfg,
        "sensor_config": sensor_cfg,
        "max_episode_steps": DEFAULT_ENV_MAX_STEPS,
        "scene_params": DEFAULT_MOUNTAIN_PARAMS_FOR_ENV,
        "mode": mode # Pass the mode to the environment
    }

    if custom_env_config: # Allow selective overrides from the calling script
        env_init_params.update(custom_env_config)

    env = CoppeliaMountainEnv(**env_init_params)
    print("\n--- Environment Initialized ---")
    print(f"Selected Mode: {env.mode}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    if env.mode == 'normal':
        print(f"Scene Params Used by Env: {env.current_scene_params}")
    print(f"Max Episode Steps (in env): {env.max_episode_steps}")
    print("-----------------------------\n")
    
    # It's a good practice to check the custom environment with SB3's checker
    # print("Running environment checker...")
    # check_env(env, warn=True) # Set warn=True to see warnings, skip_render_check=False if render works
    # print("Environment check completed.")

    return env

def run_random_agent_test(env, num_episodes=3, steps_per_episode=200):
    """Runs a simple test with a random agent."""
    print(f"\n--- Starting Random Agent Test (Mode: {env.mode}) ---")
    for episode_idx in range(1, num_episodes + 1):
        print(f"--- Test Episode {episode_idx} ---")
        obs, info = env.reset()
        # print(f"Initial observation (image shape if present): {obs.get('image', np.array([])).shape}")
        # print(f"Initial observation (joint_states): {obs.get('joint_states')}")
        total_episode_reward = 0.0
        terminated, truncated = False, False
        for step_idx in range(1, steps_per_episode + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_episode_reward += reward
            if step_idx % 50 == 0 or terminated or truncated:
                print(f"   Ep {episode_idx}, Step {env.current_step_count}: Reward={reward:.3f}, Term={terminated}, Trunc={truncated}")
            if terminated or truncated: break
        print(f"--- Test Episode {episode_idx} finished. Total Reward: {total_episode_reward:.3f} ---\n")
    # env.close() # Closing is handled in the main try/finally block

def train_with_stable_baselines3(env, total_timesteps_target=6000000, save_path_prefix="./sb3_rex_model"):
    """
    Trains an agent using Stable Baselines3.
    total_timesteps_target: The overall target number of timesteps for the entire training run.
    save_path_prefix: Prefix for saving models and checkpoints (e.g., "./sb3_rex_model").
                          The mode (e.g., 'joints_only') will be appended to this.
    """
    print(f"\n--- Starting Stable Baselines3 Training with PPO (Mode: {env.mode})") 
    
    current_mode_suffix = env.mode # e.g., 'joints_only' or 'normal'
    model_save_path = f"{save_path_prefix}_{current_mode_suffix}" # e.g., "./sb3_rex_model_joints_only"
    checkpoint_folder = f"{model_save_path}_checkpoints" # e.g., "./sb3_rex_model_joints_only_checkpoints"
    plot_save_path = f"{model_save_path}_rewards_plot.png" # Path for saving the plot

    new_learning_rate = 3e-4 # Increased learning rate
    
    # ***** USER: CHOOSE YOUR VALID CHECKPOINT STEP NUMBER HERE *****
    checkpoint_step_to_load = 1100000 # You chose 500000. Ensure this file is VALID (not 0 bytes).
                                        # If ppo_rex_500000_steps.zip is 0 bytes, pick another one, e.g., 400000
    # checkpoint_step_to_load = 400000 # Example if 500k is bad.

    # Path to the specific checkpoint file
    checkpoint_filename = f"ppo_rex_{checkpoint_step_to_load}_steps.zip"
    checkpoint_to_load_full_path = os.path.join(checkpoint_folder, checkpoint_filename)

    print(f"Attempting to load checkpoint from: {checkpoint_to_load_full_path}")

    loaded_model_num_timesteps = 0
    model_loaded_successfully = False # Flag to track if loading was successful

    if os.path.exists(checkpoint_to_load_full_path):
        # Crucially, check if the file is not empty
        if os.path.getsize(checkpoint_to_load_full_path) > 0:
            print(f"Loading pre-existing model from {checkpoint_to_load_full_path}")
            try:
                # *** THIS IS THE CORRECTED LINE ***
                model = PPO.load(checkpoint_to_load_full_path, env=env) 
                model_loaded_successfully = True # Mark as successfully loaded
                print(f"Model loaded. Current num_timesteps: {model.num_timesteps}")
                loaded_model_num_timesteps = model.num_timesteps

                print(f"Updating learning rate to {new_learning_rate}")
                if hasattr(model, 'policy') and model.policy and hasattr(model.policy, 'optimizer') and model.policy.optimizer:
                    for param_group in model.policy.optimizer.param_groups:
                        param_group['lr'] = new_learning_rate
                    print(f"Optimizer learning rate updated to {new_learning_rate}.")
                else:
                    print("Warning: Could not directly set optimizer learning rate. Ensure model.learning_rate is used if applicable.")
                    model.learning_rate = new_learning_rate
            except Exception as e:
                print(f"Error loading model from {checkpoint_to_load_full_path}: {e}")
                print("Will proceed by creating a new model.")
                model_loaded_successfully = False # Ensure flag is false on error
        else:
            print(f"Checkpoint file {checkpoint_to_load_full_path} is 0 bytes and cannot be loaded.")
            model_loaded_successfully = False
    else:
        print(f"Checkpoint {checkpoint_to_load_full_path} not found.")
        model_loaded_successfully = False # Ensure flag is false if file doesn't exist

    if not model_loaded_successfully: # If model wasn't loaded for any reason (not found, 0 bytes, load error)
        print(f"Creating a new model with learning_rate={new_learning_rate}.")
        model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=new_learning_rate)
        loaded_model_num_timesteps = 0 # Reset this as we are starting fresh

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)
        
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000, # Save every 10k steps
        save_path=checkpoint_folder,
        name_prefix="ppo_rex"
    )
    
    # Custom callback for logging rewards
    # The check_freq in RewardLoggerCallback isn't strictly necessary if we log on episode end,
    # but we keep it for potential future use or more granular logging.
    # SB3 logs 'ep_rew_mean' by default when Monitor wrapper is used or if info['episode'] is present.
    # We will rely on the info['episode'] being populated by the environment or a Monitor wrapper.
    # If your env doesn't provide this, wrap it: `env = Monitor(env)`
    # For simplicity, we'll use the new SB3 v2.0+ way of accessing episodic info if available.
    reward_logger_callback = RewardLoggerCallback(check_freq=100) # check_freq is per step here.

    timesteps_to_train_further = total_timesteps_target - loaded_model_num_timesteps
    
    if timesteps_to_train_further <= 0:
        print(f"Loaded model has {loaded_model_num_timesteps} timesteps, which is >= target {total_timesteps_target}.")
        print("Training for an additional 10,000 steps.")
        timesteps_to_train_further = 10_000 
    
    print(f"Target total timesteps: {total_timesteps_target}")
    print(f"Model current timesteps (after potential load): {loaded_model_num_timesteps}")
    print(f"Training for an additional {timesteps_to_train_further} timesteps.")

    # Pass both callbacks to the learn method
    model.learn(total_timesteps=timesteps_to_train_further, 
                callback=[checkpoint_callback, reward_logger_callback], 
                progress_bar=True, 
                reset_num_timesteps=False) # reset_num_timesteps=False is important for continued training
    
    model.save(model_save_path)
    print(f"Training complete. Model saved to {model_save_path}.zip")

    # Plotting and saving rewards
    if reward_logger_callback.rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(reward_logger_callback.timesteps, reward_logger_callback.rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(f"Episode Reward vs. Timesteps ({current_mode_suffix} mode)")
        plt.grid(True)
        plt.savefig(plot_save_path)
        print(f"Reward plot saved to {plot_save_path}")
        # plt.show() # Uncomment to display the plot
    else:
        print("No reward data logged, skipping plot generation.")


if __name__ == '__main__':
    # --- CHOOSE YOUR MODE AND SETUP ---
    # Mode can be 'normal' or 'joints_only'
    CURRENT_MODE = 'joints_only' # 'normal' or 'joints_only'
    
    # If True, uses camera sensor config. If False, uses lidar sensor config.
    USE_CAMERA = True 
    
    # Set to True to try and run CoppeliaSim without its GUI (faster for training)
    # Make sure your scene is configured to run correctly when headless.
    LAUNCH_COPPELIASIM_HEADLESS = False # Best to manage CoppeliaSim launch manually for stability

    # --- ---

    print(f"Selected run mode: {CURRENT_MODE}")
    print(f"Using camera setup: {USE_CAMERA}")
    print(f"Launch CoppeliaSim headless: {LAUNCH_COPPELIASIM_HEADLESS} (Note: Manual launch recommended)")

    # Ensure CoppeliaSim is started manually here, with the correct scene loaded.
    # For example, if USE_CAMERA is True, load "rex_camera.ttt".
    # If USE_CAMERA is False, load "rex_lidar.ttt".
    # And ensure the ZMQ remote API server is running (e.g., simRemoteApi.start(19997) in a script).
    
    # A placeholder for manual intervention if you don't have auto-launch setup
    # scene_file = DEFAULT_SCENE_CAMERA if USE_CAMERA else DEFAULT_SCENE_LIDAR
    # print_manual_intervention_instructions(scene_file)


    env_instance = None
    
    # Setup the environment with the chosen mode
    env_instance = setup_environment(
        use_camera_setup=USE_CAMERA,
        launch_headless=LAUNCH_COPPELIASIM_HEADLESS,
        mode=CURRENT_MODE,
        custom_env_config={
            "render_mode": "human" if not LAUNCH_COPPELIASIM_HEADLESS and CURRENT_MODE == 'normal' else None,
                # Add other specific overrides if needed
        }
    )
    
    if env_instance:
        # You can choose to run a random agent test or train
        # run_random_agent_test(env_instance, num_episodes=2, steps_per_episode=100)
        
        # Pass the overall target timesteps and the base path for saving
        train_with_stable_baselines3(env_instance, 
                                     total_timesteps_target=6_000_000, 
                                     save_path_prefix=f"./sb3_rex_model") # Mode will be appended by the function
