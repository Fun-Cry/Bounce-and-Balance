# env_runner.py
import gymnasium as gym
import time
import os
from env.mountain_env import CoppeliaMountainEnv
from utils.coppelia_launcher import start_coppeliasim, stop_coppeliasim # Use the launcher
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
                print(f"  Ep {episode_idx}, Step {env.current_step_count}: Reward={reward:.3f}, Term={terminated}, Trunc={truncated}")
            if terminated or truncated: break
        print(f"--- Test Episode {episode_idx} finished. Total Reward: {total_episode_reward:.3f} ---\n")
    # env.close() # Closing is handled in the main try/finally block

def train_with_stable_baselines3(env, total_timesteps=10000, save_path="./sb3_rex_model"):
    """
    Trains an agent using Stable Baselines3.
    """
    print(f"\n--- Starting Stable Baselines3 Training with PPO (Mode: {env.mode})") 
    
    # For 'joints_only' mode, if you want to be certain the image network part isn't trained,
    # you might consider freezing its weights. This is more advanced and depends on your
    # policy's feature extractor structure.
    # Example (conceptual, needs correct layer names for your MultiInputPolicy):
    # if env.mode == 'joints_only' and hasattr(model.policy, 'features_extractor'):
    #     if 'image' in model.policy.features_extractor.extractors: # For CombinedExtractor
    #         print("Attempting to freeze image feature extractor weights...")
    #         for param in model.policy.features_extractor.extractors['image'].parameters():
    #             param.requires_grad = False
    #         print("Image feature extractor weights frozen.")
    #     else:
    #         print("Could not find 'image' extractor in model.policy.features_extractor.extractors to freeze.")


    model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95)
    
    # Example of loading a pre-existing model:
    # model_load_path = save_path + ".zip"
    # if os.path.exists(model_load_path):
    #     print(f"Loading pre-existing model from {model_load_path}")
    #     model = PPO.load(model_load_path, env=env) # Make sure to pass the env or call set_env later
    # else:
    #     print("No pre-existing model found, creating a new one.")
        
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}.zip")

    # Test the trained agent (optional)
    print("\n--- Testing Trained Agent ---")
    obs, info = env.reset()
    for _ in range(DEFAULT_ENV_MAX_STEPS * 2): # Test for a couple of episodes
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render() # If you want to see it play (ensure render_mode='human' and not headless)
        if terminated or truncated:
            print(f"Trained agent episode finished. Resetting. Final reward for episode: {reward}")
            obs, info = env.reset()
            # break # Uncomment if you only want to test one episode
    print("--- Trained Agent Test Finished ---")

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
        
        train_with_stable_baselines3(env_instance, total_timesteps=25000, save_path=f"./sb3_rex_model_{CURRENT_MODE}")

