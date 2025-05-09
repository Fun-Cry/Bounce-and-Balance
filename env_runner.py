# env_runner.py
import gymnasium as gym
import time
import os
from env.mountain_env import CoppeliaMountainEnv
from utils.coppelia_launcher import start_coppeliasim, stop_coppeliasim # Use the launcher
from stable_baselines3 import PPO

# --- Default Configurations ---
DEFAULT_SCENE_CAMERA = "rex_camera.ttt"
DEFAULT_SCENE_LIDAR = "rex_lidar.ttt"
DEFAULT_ENV_MAX_STEPS = 500 # Max steps per episode for the Gym env

DEFAULT_MOUNTAIN_PARAMS_FOR_ENV = { # These are passed to CoppeliaMountainEnv's scene_params
    'n_items': 1,
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

def setup_environment(use_camera_setup=True, custom_env_config=None, launch_headless=False):
    """
    Sets up and returns the Gym environment.
    Handles one-time CoppeliaSim launch.
    """
    if use_camera_setup:
        print("CONFIG: Using CAMERA setup.")
        selected_scene = DEFAULT_SCENE_CAMERA
        sensor_cfg = {'type': 'camera', 'resolution': (64, 64)}
        render_cfg = "human" if not launch_headless else None
    else:
        print("CONFIG: Using LIDAR setup.")
        selected_scene = DEFAULT_SCENE_LIDAR
        sensor_cfg = {'type': 'lidar', 'max_points': 500}
        render_cfg = None # No human render for lidar in this basic setup

    # Attempt to launch CoppeliaSim
    if not start_coppeliasim(selected_scene, headless=launch_headless):
        print_manual_intervention_instructions(selected_scene)
    else: # If auto-launch was attempted (successfully or not, it might already be running)
        print(f"Proceeding to connect to CoppeliaSim (expected scene: '{selected_scene}').")
        # A short pause to ensure ZMQ server is fully up if just launched
        time.sleep(2)


    # Prepare parameters for CoppeliaMountainEnv initialization
    env_init_params = {
        "render_mode": render_cfg,
        "sensor_config": sensor_cfg,
        "max_episode_steps": DEFAULT_ENV_MAX_STEPS,
        "scene_params": DEFAULT_MOUNTAIN_PARAMS_FOR_ENV 
    }

    if custom_env_config: # Allow selective overrides from the calling script
        env_init_params.update(custom_env_config) # Merge/override with custom_env_config

    env = CoppeliaMountainEnv(**env_init_params)
    print("\n--- Environment Initialized ---")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Scene Params Used by Env: {env.current_scene_params}")
    print(f"Max Episode Steps (in env): {env.max_episode_steps}")
    print("-----------------------------\n")
    return env

def run_random_agent_test(env, num_episodes=3, steps_per_episode=200):
    """Runs a simple test with a random agent."""
    print("\n--- Starting Random Agent Test ---")
    for episode_idx in range(1, num_episodes + 1):
        print(f"--- Test Episode {episode_idx} ---")
        obs, info = env.reset()
        print(obs)
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
    env.close()

def train_with_stable_baselines3(env, total_timesteps=10000, save_path="./sb3_rex_model"):
    """
    Trains an agent using Stable Baselines3.
    """
    print(f"\n--- Starting Stable Baselines3 Training with PPO") 
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}.zip")

    # Test the trained agent (optional)
    print("\n--- Testing Trained Agent ---")
    obs, info = env.reset()
    for _ in range(200): # Test for 200 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            print("Trained agent episode finished and reset.")
            break
    print("--- Trained Agent Test Finished ---")

if __name__ == '__main__':
    # This allows testing env_runner.py directly
    print("Running env_runner.py as main for testing...")
    env = setup_environment(use_camera_setup=True, launch_headless=False, custom_env_config={"render_mode": None})
    run_random_agent_test(env, num_episodes=10)
    
    # env_for_sb3 = None
    # try:
    #     # For SB3, often better to run CoppeliaSim headless for speed if not debugging visually
    #     env_for_sb3 = setup_environment(use_camera_setup=True, launch_headless=True, 
    #                                      custom_env_config={"render_mode": None}) # No gym rendering for headless
    #     if env_for_sb3:
    #         train_with_stable_baselines3(env_for_sb3,  total_timesteps=20000) # Short training
    # finally:
    #     if env_for_sb3:
    #         env_for_sb3.close()
    #     stop_coppeliasim()