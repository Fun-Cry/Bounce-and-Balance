# env/mountain_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from .simulation_copp import CoppeliaSimZMQInterface # Assuming simulation_copp.py is in the same 'env' package

# Assuming actuator.py and actuator_param.py are also in the 'env' package,
# the import in simulation_copp.py `from . import actuator, actuator_param` is correct.

class DummySpring:
    def fn_spring(self, q0, q2): return 0.0, 0.0

class CoppeliaMountainEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, scene_params=None, sensor_config=None,
                 max_episode_steps=1000, mode='normal', spin_penalty_factor=0.01,
                 upward_velocity_reward_factor=10.0): # Added upward_velocity_reward_factor
        super().__init__()

        # The dt in CoppeliaSimZMQInterface's __init__ defaults to 1e-2 (0.01s).
        # This means:
        # 1. CoppeliaSim's "simulation dt" should be set to 0.01 seconds (100Hz).
        # 2. CoppeliaSim's "dynamics dt" should be <= 0.01s (e.g., 0.01s or 0.005s).
        # The self.sim_iface.dt will be 0.01 unless a different dt is passed here.
        self.sim_iface = CoppeliaSimZMQInterface(spring=DummySpring()) # Uses default dt=1e-2 from simulation_copp.py
        
        self.render_mode = render_mode
        self.sensor_config = sensor_config if sensor_config else {'type': 'camera', 'resolution': (64, 64)}
        self.mode = mode
        self.spin_penalty_factor = spin_penalty_factor
        self.upward_velocity_reward_factor = upward_velocity_reward_factor

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        obs_dict = {}
        if self.sensor_config['type'] == 'camera':
            res = self.sensor_config.get('resolution', (64,64))
            obs_dict["image"] = spaces.Box(low=0,high=255,shape=(res[1],res[0],3),dtype=np.uint8)
        elif self.sensor_config['type'] == 'lidar':
            pts = self.sensor_config.get('max_points', 500)
            obs_dict["lidar_points"] = spaces.Box(low=-np.inf,high=np.inf,shape=(pts,3),dtype=np.float32)
        else: raise ValueError(f"Unsupported sensor_type: {self.sensor_config['type']}")
        
        # Assuming 7 joints: Joint_0, Joint_1, Joint_2, Joint_3, joint_rw0, joint_rw1, joint_rwz
        # So 7 positions + 7 velocities = 14 states
        obs_dict["joint_states"] = spaces.Box(low=-np.inf,high=np.inf,shape=(14,),dtype=np.float32)
        obs_dict["imu_data"] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32) # 3 linear vel + 3 angular vel
        self.observation_space = spaces.Dict(obs_dict)

        self.default_scene_params = {
            'n_items':1,'shape_options':9,'mountain_target_total_height':1.0,
            'mountain_max_cylinder_height':0.11,'mountain_min_cylinder_height':0.05,
            'mountain_peak_radius':0.3,'mountain_base_radius_factor_range':(3.0,5.0),
            'mountain_area_bounds_x':(-2.5,2.5),'mountain_area_bounds_y':(1.5,3.5)
        }
        self.current_scene_params = scene_params if scene_params is not None else self.default_scene_params
        
        self.window_name = "CoppeliaSim Rex Env"
        if self.render_mode=="human" and self.sensor_config['type']=='camera': cv2.namedWindow(self.window_name)
        
        self.max_episode_steps = max_episode_steps
        self.current_step_count = 0
        self.fall_height_threshold = 0.1 # Adjusted based on typical robot scale
        self.fall_angle_threshold = np.deg2rad(60) # Radians

        if self.mode == 'joints_only':
            print("INFO: Environment initialized in 'joints_only' mode. Image data will be zeroed, and no scenery will be spawned.")


    def _get_observation(self):
        rgb, lidar = self.sim_iface.get_sensor_data()
        j_pos, j_vel = self.sim_iface._get_joint_states()
        base_lin_vel, base_ang_vel = self.sim_iface.get_base_imu_data()
        obs = {}

        if self.sensor_config['type'] == 'camera':
            res = self.sensor_config.get('resolution', (64,64))
            if self.mode == 'joints_only':
                img = np.zeros((res[1], res[0], 3), dtype=np.uint8)
            elif rgb is None:
                img = np.zeros((res[1], res[0], 3), dtype=np.uint8)
            elif rgb.shape[:2] != (res[1], res[0]):
                img = cv2.resize(rgb, res, interpolation=cv2.INTER_AREA)
            else:
                img = rgb
            obs["image"] = img
        elif self.sensor_config['type'] == 'lidar':
            pts = self.sensor_config.get('max_points',500)
            if lidar is None or lidar.shape[0]==0:
                l_pts = np.zeros((pts,3),dtype=np.float32)
            else:
                num_avail = lidar.shape[0]
                if num_avail>=pts:
                    l_pts = lidar[:pts,:]
                else:
                    l_pts = np.vstack((lidar, np.zeros((pts-num_avail,3),dtype=np.float32)))
            obs["lidar_points"] = l_pts
        
        # Ensure all expected joint aliases are present for consistent observation shape
        aliases = ['/Joint_0','/Joint_1','/Joint_2','/Joint_3','/joint_rw0','/joint_rw1','/joint_rwz']
        pos_arr = np.array([j_pos.get(k,0.0) for k in aliases],dtype=np.float32)
        vel_arr = np.array([j_vel.get(k,0.0) for k in aliases],dtype=np.float32)
        obs["joint_states"] = np.concatenate([pos_arr, vel_arr])
        obs["imu_data"] = np.concatenate([base_lin_vel, base_ang_vel])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        params = self.default_scene_params
        if options and "scene_params" in options:
            params = options["scene_params"]
        elif self.current_scene_params is not self.default_scene_params:
            params = self.current_scene_params
        
        if self.mode == 'joints_only':
            self.sim_iface.reset_environment(sequential=True) # No scenery spawned
        else:
            self.sim_iface.reset_environment(**params)

        self.current_step_count = 0
        obs = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        self.sim_iface.control(action) # This internally calls self.sim.step()
        self.current_step_count+=1
        obs = self._get_observation()
        reward = 0.0 # Initialize reward for the step
        terminated = False
        truncated = False
        
        current_height = 0.0 # Default height if not available
        upward_velocity_bonus = 0.0
        spin_penalty = 0.0

        if self.sim_iface.robot_base:
            try:
                pos = self.sim_iface.sim.getObjectPosition(self.sim_iface.robot_base, -1) # Use world frame
                current_height = float(pos[2])
                reward = current_height # Base reward is height

                orient_euler = self.sim_iface.sim.getObjectOrientation(self.sim_iface.robot_base, -1) # Euler angles in radians
                
                base_lin_vel, base_ang_vel = self.sim_iface.get_base_imu_data()
                
                # Spin penalty
                # Penalize angular velocity around the world Z-axis, or overall spin
                # base_ang_vel[2] is yaw rate if IMU is aligned with world frame when robot is flat
                # Using magnitude for general spin penalty
                spin_magnitude = np.linalg.norm(base_ang_vel) 
                spin_penalty = self.spin_penalty_factor * spin_magnitude
                reward -= spin_penalty

                # Upward velocity reward
                upward_velocity_z = base_lin_vel[2] # Z-component of linear velocity in world frame
                if upward_velocity_z > 0:
                    upward_velocity_bonus = self.upward_velocity_reward_factor * upward_velocity_z
                    reward += upward_velocity_bonus
                
                # Termination conditions
                # Fall detection based on orientation (roll and pitch angles)
                tilted_too_much = abs(orient_euler[0]) > self.fall_angle_threshold or \
                                  abs(orient_euler[1]) > self.fall_angle_threshold
                
                # Fall detection based on height (relative to initial or an absolute minimum)
                # Assuming initial_base_pos[2] is the starting height on flat ground.
                # Or use a fixed low threshold if starting on uneven terrain.
                # For simplicity, using self.fall_height_threshold as an absolute minimum.
                fallen_too_low = current_height < self.fall_height_threshold 

                if tilted_too_much or fallen_too_low:
                    terminated = True
                    reward = -200.0 # Significant penalty for falling
            except Exception as e:
                # print(f"Warning: Could not compute reward components: {e}") # Optional for debugging
                pass # Keep going if minor sim error
        
        if not terminated and self.current_step_count >= self.max_episode_steps:
            truncated = True
        
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode in ["human","rgb_array"] and self.sensor_config['type']=='camera':
            img_obs = self._get_observation() 
            img = img_obs.get("image")
            if img is not None:
                if self.render_mode=="human":
                    # Ensure window exists before trying to show image
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                         cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE) # Recreate if closed
                    cv2.imshow(self.window_name,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                else: # rgb_array
                    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # Return BGR for consistency if other tools expect it
        return None

    def close(self):
        self.sim_iface.close() # This should stop simulation and release ZMQ client
        if self.render_mode=="human" and self.sensor_config['type']=='camera':
            # Check if window exists before trying to destroy
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(self.window_name)
            cv2.waitKey(1) # Allow time for CV2 to process window closure