# test_checkpoint.py
import gymnasium as gym
import time
import os
import argparse
from env.mountain_env import CoppeliaMountainEnv # Assuming this path is correct
from stable_baselines3 import PPO
# from utils.coppelia_launcher import start_coppeliasim, stop_coppeliasim # If needed

# --- Default Configurations (can be overridden by args) ---
DEFAULT_SCENE_CAMERA = "rex_camera.ttt"
DEFAULT_SCENE_LIDAR = "rex_lidar.ttt"
DEFAULT_ENV_MAX_STEPS = 500 # Max steps per episode for the Gym env

DEFAULT_MOUNTAIN_PARAMS_FOR_ENV = { # These are passed to CoppeliaMountainEnv's scene_params
    'n_items': 1, # Number of mountains/obstacles
    'mountain_peak_radius': 0.35,
    'mountain_base_radius_factor_range': (2.5, 4.5),
    # Other params will use CoppeliaMountainEnv's internal defaults if not specified
}

def setup_testing_environment(use_camera_setup=True, custom_env_config=None, mode='normal', render_human=False, raw=False):
    """
    Sets up and returns the Gym environment for testing.
    'mode' can be 'normal' or 'joints_only'.
    'render_human' determines if 'human' render_mode is used.
    """
    print(f"--- Setting up Environment for Testing ---")
    print(f"CONFIG: Mode: {mode}, Use Camera: {use_camera_setup}, Render Human: {render_human}")

    if use_camera_setup:
        # selected_scene = DEFAULT_SCENE_CAMERA # For manual instruction or auto-launch
        sensor_cfg = {'type': 'camera', 'resolution': (64, 64)}
    else: # Lidar setup
        # selected_scene = DEFAULT_SCENE_LIDAR # For manual instruction or auto-launch
        sensor_cfg = {'type': 'lidar', 'max_points': 500}

    render_cfg = "human" if render_human else None

    # --- Manual CoppeliaSim Launch Reminder ---
    # scene_to_remind = selected_scene
    # print(f"Reminder: Ensure CoppeliaSim is running with the appropriate scene ('{scene_to_remind}') and ZMQ server enabled.")
    # print_manual_intervention_instructions(scene_to_remind) # Uncomment if you want this prompt

    print(f"Attempting to connect to CoppeliaSim. Please ensure it's running with the appropriate scene and ZMQ server.")

    env_init_params = {
        "render_mode": render_cfg,
        "sensor_config": sensor_cfg,
        "max_episode_steps": DEFAULT_ENV_MAX_STEPS,
        "scene_params": DEFAULT_MOUNTAIN_PARAMS_FOR_ENV,
        "mode": mode,
        "raw": raw
    }

    if custom_env_config:
        env_init_params.update(custom_env_config)

    env = CoppeliaMountainEnv(**env_init_params)
    print("\n--- Test Environment Initialized ---")
    print(f"Selected Mode: {env.mode}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    if env.mode == 'normal':
        print(f"Scene Params Used by Env: {env.current_scene_params}")
    print(f"Max Episode Steps (in env): {env.max_episode_steps}")
    print(f"Render Mode: {env.render_mode}")
    print("----------------------------------\n")
    
    # from stable_baselines3.common.env_checker import check_env # Optional: Check if needed
    # print("Running environment checker (optional)...")
    # check_env(env, warn=True, skip_render_check= not render_human)
    # print("Environment check completed.")

    return env

def test_trained_model(env, model_path, num_episodes=100, deterministic=True):
    """
    Loads a trained SB3 model and tests it in the environment.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found.")
        return

    print(f"\n--- Starting Trained Model Test ---")
    print(f"Loading model from: {model_path}")
    print(f"Number of test episodes: {num_episodes}")
    print(f"Deterministic actions: {deterministic}")
    print(f"Environment mode: {env.mode}")

    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    total_rewards_all_episodes = []

    for episode_idx in range(1, num_episodes + 1):
        print(f"\n--- Test Episode {episode_idx} ---")
        obs, info = env.reset()
        terminated, truncated = False, False
        total_episode_reward = 0.0
        current_episode_steps = 0

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_episode_reward += reward
            current_episode_steps += 1

            if env.render_mode == "human":
                env.render() # Call render if in human mode
                time.sleep(0.01) # Small delay to make rendering viewable

            if current_episode_steps % 50 == 0:
                 print(f"  Ep {episode_idx}, Step {current_episode_steps}: Last Reward={reward:.3f}, Term={terminated}, Trunc={truncated}")

            if terminated:
                print(f"  Episode {episode_idx} terminated after {current_episode_steps} steps.")
            if truncated:
                print(f"  Episode {episode_idx} truncated after {current_episode_steps} steps (max steps reached).")

        print(f"--- Test Episode {episode_idx} finished. Total Reward: {total_episode_reward:.3f} ---")
        total_rewards_all_episodes.append(total_episode_reward)

    print("\n--- Trained Model Test Finished ---")
    if total_rewards_all_episodes:
        avg_reward = sum(total_rewards_all_episodes) / len(total_rewards_all_episodes)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.3f}")
        print(f"Individual episode rewards: {total_rewards_all_episodes}")
    print("-----------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained Stable Baselines3 PPO agent for CoppeliaMountainEnv.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the SB3 model checkpoint (.zip file).")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to test the agent.")
    parser.add_argument("--mode", type=str, default="joints_only", choices=["normal", "joints_only"], help="Environment mode ('normal' or 'joints_only').")
    parser.add_argument("--use_camera", action=argparse.BooleanOptionalAction, default=True, help="Use camera sensor setup (default). Use --no-use_camera for lidar.")
    parser.add_argument("--render", action="store_true", help="Enable human rendering of the environment.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Use deterministic actions for prediction.")
    parser.add_argument("--raw", action=argparse.BooleanOptionalAction, default=False, help="Use deterministic actions for prediction.")
    parser.add_argument("--dt", type=float, default=5e-2, help="Number of episodes to test the agent.")
    # parser.add_argument("--max_steps_per_episode", type=int, default=DEFAULT_ENV_MAX_STEPS, help="Override max steps per episode for testing.") # Optional override

    args = parser.parse_args()

    # --- Manual CoppeliaSim Launch Reminder ---
    # This is important as the script doesn't auto-launch CoppeliaSim.
    scene_file_name = DEFAULT_SCENE_CAMERA if args.use_camera else DEFAULT_SCENE_LIDAR
    # print_manual_intervention_instructions(scene_file_name) # Remind user to prepare CoppeliaSims

    env_instance = None
    try:
        # Custom config for environment if needed, e.g., to override max_episode_steps for testing
        # custom_config_for_test = {"max_episode_steps": args.max_steps_per_episode}
        custom_config_for_test = {} # No overrides by default other than render_mode

        env_instance = setup_testing_environment(
            use_camera_setup=args.use_camera,
            mode=args.mode,
            render_human=args.render,
            custom_env_config=custom_config_for_test,
            raw=args.raw
        )

        if env_instance:
            test_trained_model(
                env=env_instance,
                model_path=args.checkpoint_path,
                num_episodes=args.num_episodes,
                deterministic=args.deterministic
            )

    except Exception as e:
        print(f"An error occurred during the testing script: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env_instance:
            print("Closing environment...")
            env_instance.close()
            print("Environment closed.")
        # stop_coppeliasim() # If you were using the launcher

    print("Testing script finished.")