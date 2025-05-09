# start_training.py
from env_runner import run_env_session, stop_coppeliasim_instance # Import new functions

def main():
    """
    Clean and simple interface to start an environment run.
    CoppeliaSim will be launched once by env_runner if not already running from this script.
    """
    print("Initiating Rex Environment Run Session...")

    # --- High-Level Configuration for this Run ---
    USE_CAMERA_SENSOR = True  # True for camera, False for LiDAR
    NUMBER_OF_EPISODES = 2
    STEPS_IN_EXAMPLE_LOOP = 250 
    LAUNCH_COPPELIASIM_HEADLESS = False # For the CoppeliaSim application itself

    # Optional: Provide custom parameters to override defaults
    custom_environment_parameters = {
        "scene_params": {
            'n_items': 1,
            'mountain_max_cylinder_height': 0.15,
        },
        "max_episode_steps": 150, # Max steps for truncation within the CoppeliaMountainEnv instance
        # "render_mode": "human" # env_runner now deduces this from auto_launch_headless for camera
    }
    
    try:
        run_env_session(
            use_camera_setup=USE_CAMERA_SENSOR,
            custom_env_params=custom_environment_parameters,
            num_episodes=NUMBER_OF_EPISODES,
            num_steps_per_episode_loop=STEPS_IN_EXAMPLE_LOOP,
            auto_launch_headless=LAUNCH_COPPELIASIM_HEADLESS
        )
    finally:
        # This ensures the CoppeliaSim instance launched by env_runner is stopped
        # when the main script finishes or if an error occurs.
        stop_coppeliasim_instance() 

    print("Rex Environment Run Session script finished.")

if __name__ == '__main__':
    main()