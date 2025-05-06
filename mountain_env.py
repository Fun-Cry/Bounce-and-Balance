# mountain_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

from env.simulation_copp import CoppeliaSimZMQInterface # Your existing interface

class DummySpring: # Placeholder
    def fn_spring(self, q0, q2):
        return 0.0, 0.0

class CoppeliaMountainEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, scene_params=None,
                 sensor_config=None):
        super().__init__()

        self.sim_iface = CoppeliaSimZMQInterface(spring=DummySpring())
        self.render_mode = render_mode

        if sensor_config is None:
            self.sensor_config = {'type': 'camera', 'resolution': (64, 64)}
        else:
            self.sensor_config = sensor_config

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        obs_space_dict = {}
        if self.sensor_config['type'] == 'camera':
            resolution = self.sensor_config.get('resolution', (64, 64))
            obs_space_dict["image"] = spaces.Box(
                low=0, high=255,
                shape=(resolution[1], resolution[0], 3), # H, W, C
                dtype=np.uint8
            )
        elif self.sensor_config['type'] == 'lidar':
            max_points = self.sensor_config.get('max_points', 500)
            obs_space_dict["lidar_points"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_points, 3),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported sensor_type: {self.sensor_config['type']}")

        num_joint_states = 14
        obs_space_dict["joint_states"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_joint_states,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(obs_space_dict)

        self.default_scene_params = {
            'n_items': 4, 'shape_options': 9, 'mountain_target_total_height': 1.0,
            'mountain_max_cylinder_height': 0.2, 'mountain_min_cylinder_height': 0.05,
            'mountain_peak_radius': 0.25, 'mountain_base_radius_factor_range': (3.0, 6.0),
            'mountain_area_bounds_x': (-2.5, 2.5), 'mountain_area_bounds_y': (1.5, 3.5)
        }
        self.current_scene_params = scene_params if scene_params is not None else self.default_scene_params
        
        self.window_name = "CoppeliaSim Mountain Env"
        if self.render_mode == "human" and self.sensor_config['type'] == 'camera':
             cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def _get_observation(self):
        raw_rgb_image, raw_lidar_data = self.sim_iface.get_sensor_data(show=False)
        joint_pos, joint_vel = self.sim_iface._get_joint_states()
        obs_data = {}

        if self.sensor_config['type'] == 'camera':
            resolution = self.sensor_config.get('resolution', (64, 64))
            if raw_rgb_image is None:
                processed_image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            else:
                if raw_rgb_image.shape[0] != resolution[1] or raw_rgb_image.shape[1] != resolution[0]:
                    processed_image = cv2.resize(raw_rgb_image, resolution, interpolation=cv2.INTER_AREA)
                else:
                    processed_image = raw_rgb_image
            obs_data["image"] = processed_image
        elif self.sensor_config['type'] == 'lidar':
            max_points = self.sensor_config.get('max_points', 500)
            if raw_lidar_data is None or raw_lidar_data.shape[0] == 0:
                processed_lidar = np.zeros((max_points, 3), dtype=np.float32)
            else:
                num_available_points = raw_lidar_data.shape[0]
                if num_available_points >= max_points:
                    processed_lidar = raw_lidar_data[:max_points, :]
                else:
                    padding = np.zeros((max_points - num_available_points, 3), dtype=np.float32)
                    processed_lidar = np.vstack((raw_lidar_data, padding))
            obs_data["lidar_points"] = processed_lidar

        joint_aliases_ordered = ['/Joint_0','/Joint_1','/Joint_2','/Joint_3',
                                 '/joint_rw0','/joint_rw1','/joint_rwz']
        positions = np.array([joint_pos.get(alias, 0.0) for alias in joint_aliases_ordered], dtype=np.float32)
        velocities = np.array([joint_vel.get(alias, 0.0) for alias in joint_aliases_ordered], dtype=np.float32)
        obs_data["joint_states"] = np.concatenate([positions, velocities])
        return obs_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scene_params" in options:
            self.current_scene_params = options["scene_params"]
        self.sim_iface.reset_environment(**self.current_scene_params)
        observation = self._get_observation()
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        # Apply action
        self.sim_iface.control(action) # This internally calls sim.step()
        
        # Get new observation
        observation = self._get_observation()
        
        # --- MODIFIED: Reward Function ---
        reward = 0.0
        if self.sim_iface.robot_base is not None:
            try:
                # Get the position of the robot's base link in world coordinates
                base_position = self.sim_iface.sim.getObjectPosition(self.sim_iface.robot_base, -1) # -1 for world frame
                # The Z-coordinate (index 2) is the height
                robot_height = float(base_position[2])
                reward = robot_height
            except Exception as e:
                print(f"Warning: Could not get robot base position for reward: {e}")
                reward = 0.0 # Default to 0 if there's an issue
        else:
            # If robot_base handle was never found, reward is 0
            reward = 0.0
        # --- End of Reward Function ---

        # --- Termination and Truncation (Placeholders - you'll need to define these) ---
        terminated = False # E.g., if robot falls over, reaches a specific height, or goes out of bounds
        truncated = False  # E.g., if episode reaches a maximum step limit

        # Example termination: if robot falls (e.g. Z coordinate too low, or orientation too tilted)
        # This is just an example, you might need more robust fall detection
        if 'base_position' in locals() and base_position[2] < 0.05: # Assuming ground is around Z=0
            # terminated = True
            # reward -= 10 # Add a penalty for falling, adjust as needed
            pass

        # Example truncation:
        # if self.current_step_count >= self.max_episode_steps:
        #    truncated = True
            
        info = {} # Add any auxiliary diagnostic information

        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        # ... (render method remains the same as previously provided) ...
        if self.render_mode == "human" and self.sensor_config['type'] == 'camera':
            obs_dict = self._get_observation() 
            rgb_image = obs_dict.get("image")
            if rgb_image is not None:
                cv2.imshow(self.window_name, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                return None
        elif self.render_mode == "rgb_array" and self.sensor_config['type'] == 'camera':
            obs_dict = self._get_observation()
            return obs_dict.get("image")
        return None


    def close(self):
        # ... (close method remains the same as previously provided) ...
        self.sim_iface.close()
        if self.render_mode == "human" and self.sensor_config['type'] == 'camera':
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                 cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)

# Example usage (optional, for testing the environment)
if __name__ == '__main__':
    # --- OPTION 1: CAMERA ENVIRONMENT ---
    print("Testing CAMERA environment configuration with height reward...")
    # IMPORTANT: Manually start CoppeliaSim with 'rex_camera.ttt' before running this.
    
    camera_sensor_config = {'type': 'camera', 'resolution': (64, 64)}
    # Example scene parameters (can be customized)
    scene_config = {
        'n_items': 3,
        'mountain_target_total_height': 0.7,
        'mountain_max_cylinder_height': 0.1,
        'mountain_peak_radius': 0.35,
        'mountain_base_radius_factor_range': (2.5, 4.0)
    }
    env_camera = CoppeliaMountainEnv(render_mode="human", 
                                     sensor_config=camera_sensor_config,
                                     scene_params=scene_config)
    obs, info = env_camera.reset()
    print("Camera Env - Initial Obs (image shape):", obs["image"].shape)
    
    total_reward = 0
    for i in range(50): # Test for 50 steps
        action = env_camera.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env_camera.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action={action}, Reward (Height)={reward:.3f}, Terminated={terminated}, Truncated={truncated}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps.")
            break
    print(f"Total reward over test steps: {total_reward:.3f}")
    env_camera.close()
    print("Camera environment test finished.\n")