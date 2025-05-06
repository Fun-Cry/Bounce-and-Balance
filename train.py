# Your main training script (e.g., train_agent.py)
import gymnasium as gym
from mountain_env import CoppeliaMountainEnv # From the file above
import subprocess # To potentially help launch CoppeliaSim
import time
import os

# --- Configuration ---
USE_CAMERA = True # Set to False to use LiDAR
COPPELIASIM_PATH = "/path/to/your/coppeliasim_folder" # e.g., "/home/user/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04"
SCENE_CAMERA = "rex_camera.ttt" # Relative to CoppeliaSim path or absolute
SCENE_LIDAR = "rex_lidar.ttt"   # Relative to CoppeliaSim path or absolute
HEADLESS_MODE = True # Run CoppeliaSim with -h

# --- Helper to launch CoppeliaSim (Optional) ---
def launch_coppeliasim(scene_file_name, headless=True):
    coppeliasim_executable = os.path.join(COPPELIASIM_PATH, "coppeliaSim.sh") # For Linux
    # Adjust for coppeliasim.exe on Windows or .app on macOS
    
    cmd = [coppeliasim_executable]
    if headless:
        cmd.append("-h")
    
    # Assuming scene files are in the CoppeliaSim root directory or you provide full path
    # For simplicity, let's assume they are in a 'scenes' subfolder of COPPELIASIM_PATH
    # You might need to adjust this path logic.
    scene_path = os.path.join(COPPELIASIM_PATH, scene_file_name) # Or your specific path logic
    if not os.path.exists(scene_path):
         # Fallback if not in root, try common 'scenes' subdir
         scene_path_alt = os.path.join(COPPELIASIM_PATH, "scenes", scene_file_name)
         if os.path.exists(scene_path_alt):
             scene_path = scene_path_alt
         else:
             print(f"Warning: Scene file {scene_file_name} not found at {scene_path} or in 'scenes' subdir.")
             # Proceeding, assuming user launched manually or scene is in default search paths
    
    cmd.append(scene_path) # Add scene file to command
    
    print(f"Attempting to launch CoppeliaSim with command: {' '.join(cmd)}")
    # Important: CoppeliaSim should run in the background or a separate process.
    # The ZMQ server needs to start. The Python script will then connect.
    # Using Popen for non-blocking start.
    process = subprocess.Popen(cmd)
    print(f"CoppeliaSim process started with PID: {process.pid}. Waiting for ZMQ server to initialize...")
    time.sleep(10) # Give CoppeliaSim time to load and start ZMQ server (adjust as needed)
    return process


# --- Main training logic ---
if __name__ == '__main__':
    coppeliasim_process = None
    try:
        if USE_CAMERA:
            print("Configuring for CAMERA scene.")
            selected_scene_file = SCENE_CAMERA
            sensor_config = {'type': 'camera', 'resolution': (64, 64)}
            # coppeliasim_process = launch_coppeliasim(selected_scene_file, headless=HEADLESS_MODE) # Optional: auto-launch
        else:
            print("Configuring for LIDAR scene.")
            selected_scene_file = SCENE_LIDAR
            sensor_config = {'type': 'lidar', 'max_points': 500}
            # coppeliasim_process = launch_coppeliasim(selected_scene_file, headless=HEADLESS_MODE) # Optional: auto-launch

        print(f"IMPORTANT: Please ensure CoppeliaSim is running with '{selected_scene_file}'")
        print("and its ZMQ Remote API server is started (typically on port 19997 or as configured in scene).")
        input("Press Enter to continue once CoppeliaSim is ready...")


        # Define mountain/scene generation parameters
        custom_scene_params = {
            'n_items': 3,
            'mountain_max_cylinder_height': 0.15,
            'mountain_base_radius_factor_range': (2.0, 4.0),
            # Add other scene parameters as defined in CoppeliaSimZMQInterface.spawn_scenery
        }

        env = CoppeliaMountainEnv(
            render_mode="human" if USE_CAMERA and not HEADLESS_MODE else None, # Only render if camera and not fully headless training
            sensor_config=sensor_config,
            scene_params=custom_scene_params
        )

        obs, info = env.reset()
        print(f"Environment for {sensor_config['type']} reset successfully.")
        print("Observation space:", env.observation_space)
        # print("Initial observation keys:", obs.keys())

        for episode in range(3): # Example: 3 episodes
            obs, info = env.reset()
            done = False
            truncated = False
            step_count = 0
            print(f"--- Episode {episode + 1} ---")
            while not (done or truncated):
                action = env.action_space.sample()  # Your agent's action here
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
                if step_count % 50 == 0:
                    print(f"Step {step_count}, Reward: {reward}")
                if step_count > 200: # Max steps per episode
                    truncated = True
            print(f"Episode finished after {step_count} steps.")

        env.close()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if coppeliasim_process:
            print(f"Terminating CoppeliaSim process (PID: {coppeliasim_process.pid})...")
            coppeliasim_process.terminate()
            coppeliasim_process.wait()
        print("Training script finished.")