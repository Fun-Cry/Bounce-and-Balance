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
                 max_episode_steps=1000, mode='normal', spin_penalty_factor: float = 10.0,
                 angle_max_reward: float = 100.0,
                 angle_activation_deg: float = 30.0,
                 angle_min_vz_threshold: float = 0.01,
                 angle_min_speed_threshold: float = 0.05,
                 downward_straightness_reward_scale: float = 1.0,
                 # --- Parameters for MINIMAL "Land Well" reward ---
                 land_impact_vz_penalty_scale: float = 0.5, 
                 land_impact_upright_bonus: float = 20.0,  
                 land_impact_max_tilt_deg: float = 25.0,   
                 land_detection_falling_vz_thresh: float = -0.1, 
                 land_detection_landed_vz_thresh: float = -0.05, 
                 land_detection_height_offset: float = 0.03,
                 # --- Parameters for Link Spread Straightness ---
                 max_link_spread_x: float = 0.2, # Max allowed X-distance between Link_2 and Link_3 (meters)
                 max_link_spread_y: float = 0.2, # Max allowed Y-distance between Link_2 and Link_3 (meters)
                 link_spread_penalty_factor: float = 5.0, # Penalty factor for exceeding link spread thresholds
                 # --- Parameters for Jump Count Reward ---
                 jump_count_height_threshold: float = 0.5,
                 jump_count_reward_value: float = 50.0,
                 jump_count_reward_weight: float = 1.0
                 ):
        super().__init__()

        self.sim_iface = CoppeliaSimZMQInterface(spring=DummySpring())
        
        self.render_mode = render_mode
        self.sensor_config = sensor_config if sensor_config else {'type': 'camera', 'resolution': (64, 64)}
        self.mode = mode
        
        # Reward parameters
        self.spin_penalty_factor = spin_penalty_factor
        self.angle_max_reward = angle_max_reward
        self.angle_activation_deg = angle_activation_deg
        self.angle_min_vz_threshold = angle_min_vz_threshold
        self.angle_min_speed_threshold = angle_min_speed_threshold
        self.downward_straightness_reward_scale = downward_straightness_reward_scale

        # --- Store MINIMAL "Land Well" parameters ---
        self.land_impact_vz_penalty_scale = land_impact_vz_penalty_scale 
        self.land_impact_upright_bonus = land_impact_upright_bonus
        self.land_impact_max_tilt_rad = np.deg2rad(land_impact_max_tilt_deg) 
        self.land_detection_falling_vz_thresh = land_detection_falling_vz_thresh
        self.land_detection_landed_vz_thresh = land_detection_landed_vz_thresh
        self.land_detection_height_offset = land_detection_height_offset

        # --- Store Link Spread Straightness parameters ---
        self.max_link_spread_x = max_link_spread_x
        self.max_link_spread_y = max_link_spread_y
        self.link_spread_penalty_factor = link_spread_penalty_factor

        # --- Store Jump Count Reward parameters ---
        self.jump_count_height_threshold = jump_count_height_threshold
        self.jump_count_reward_value = jump_count_reward_value
        self.jump_count_reward_weight = jump_count_reward_weight
        self._was_above_jump_count_threshold = False


        # --- Get handles for specified links for spread calculation ---
        self.link2_name = "/Link_2_respondable"
        self.link3_name = "/Link_3_respondable"
        self.link2_handle = None
        self.link3_handle = None

        if self.sim_iface and self.sim_iface.sim:
            try:
                self.link2_handle = self.sim_iface.sim.getObject(self.link2_name)
                if self.link2_handle == -1: # getObject returns -1 if not found
                    print(f"Warning: CoppeliaMountainEnv __init__ - Could not get handle for {self.link2_name}. Spread calculation will be affected.")
                    self.link2_handle = None 
            except Exception as e:
                print(f"Error: CoppeliaMountainEnv __init__ - Exception getting handle for {self.link2_name}: {e}. Spread calculation will be affected.")
                self.link2_handle = None
            
            try:
                self.link3_handle = self.sim_iface.sim.getObject(self.link3_name)
                if self.link3_handle == -1: # getObject returns -1 if not found
                    print(f"Warning: CoppeliaMountainEnv __init__ - Could not get handle for {self.link3_name}. Spread calculation will be affected.")
                    self.link3_handle = None
            except Exception as e:
                print(f"Error: CoppeliaMountainEnv __init__ - Exception getting handle for {self.link3_name}: {e}. Spread calculation will be affected.")
                self.link3_handle = None
        else:
            print("Warning: CoppeliaMountainEnv __init__ - sim_iface or sim_iface.sim not available for link handle retrieval.")


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        obs_dict = {}
        if self.sensor_config['type'] == 'camera':
            res = self.sensor_config.get('resolution', (64,64))
            obs_dict["image"] = spaces.Box(low=0,high=255,shape=(res[1],res[0],3),dtype=np.uint8)
        elif self.sensor_config['type'] == 'lidar':
            pts = self.sensor_config.get('max_points', 500)
            obs_dict["lidar_points"] = spaces.Box(low=-np.inf,high=np.inf,shape=(pts,3),dtype=np.float32)
        else: raise ValueError(f"Unsupported sensor_type: {self.sensor_config['type']}")
        
        obs_dict["joint_states"] = spaces.Box(low=-np.inf,high=np.inf,shape=(14,),dtype=np.float32)
        obs_dict["imu_data"] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
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
        self.fall_height_threshold = 0.2 
        self.fall_angle_threshold_rad = np.deg2rad(60) 

        self.previous_base_lin_vel_z = 0.0

        if self.mode == 'joints_only':
            print("INFO: Environment initialized in 'joints_only' mode. Image data will be zeroed, and no scenery will be spawned.")

    # --- Individual Reward Component Methods ---

    def _calculate_spin_penalty(self, base_angular_velocity: np.ndarray) -> float:
        """Calculates the penalty for spinning."""
        return -self.spin_penalty_factor * np.linalg.norm(base_angular_velocity)

    def _calculate_upward_direction_reward(self, linear_velocity_xyz: np.ndarray) -> float:
        """Calculates reward for moving upwards, aligned with the vertical axis."""
        movement_direction = np.array(linear_velocity_xyz, dtype=np.float32)
        vertical_velocity_component = movement_direction[2] 
        norm_movement = np.linalg.norm(movement_direction)

        if norm_movement < self.angle_min_speed_threshold: return 0.0
        if vertical_velocity_component < self.angle_min_vz_threshold: return 0.0
        if norm_movement == 0: return 0.0 

        cos_angle = np.clip(vertical_velocity_component / norm_movement, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        if angle_deg <= self.angle_activation_deg:
            scaled_angle_rad_for_cosine = (angle_deg / self.angle_activation_deg) * (np.pi / 2.0)
            reward_val = self.angle_max_reward * np.cos(scaled_angle_rad_for_cosine)
            return max(0.0, reward_val) 
        return 0.0

    def _calculate_downward_straightness_reward(self, orientation_euler_rad: np.ndarray, current_vz: float) -> float:
        """
        Calculates reward for descending straight down.
        Combines two aspects:
        1. Minimal tilt of the base link.
        2. Minimal horizontal spread between Link_2_respondable and Link_3_respondable.
        """
        reward = 0.0
        if current_vz < 0.0: # Only apply if moving downwards
            # 1. Tilt-based straightness (from base link orientation)
            tilt_rad = np.sqrt(orientation_euler_rad[0]**2 + orientation_euler_rad[1]**2)
            straightness_from_tilt = 0.0
            if tilt_rad < self.fall_angle_threshold_rad:
                straightness_from_tilt = 1.0 - (tilt_rad / self.fall_angle_threshold_rad)

            # 2. Link Spread (horizontal distance between Link_2 and Link_3)
            straightness_from_link_spread = 1.0 # Default to no penalty if positions cannot be obtained or links not found

            if self.sim_iface and self.sim_iface.sim and self.link2_handle is not None and self.link3_handle is not None:
                sim_api = self.sim_iface.sim
                
                pos_link2 = sim_api.getObjectPosition(self.link2_handle, -1) # -1 for world frame
                pos_link3 = sim_api.getObjectPosition(self.link3_handle, -1) # -1 for world frame

                if (pos_link2 and isinstance(pos_link2, (list, tuple)) and len(pos_link2) == 3 and
                    pos_link3 and isinstance(pos_link3, (list, tuple)) and len(pos_link3) == 3):
                    
                    # Calculate horizontal spread (distance) between the two links
                    spread_x = abs(pos_link2[0] - pos_link3[0])
                    spread_y = abs(pos_link2[1] - pos_link3[1])
                    
                    # Calculate penalty for exceeding spread thresholds
                    spread_x_excess = max(0, spread_x - self.max_link_spread_x)
                    spread_y_excess = max(0, spread_y - self.max_link_spread_y)
                    
                    total_spread_excess = spread_x_excess + spread_y_excess
                    # Score is 1.0 if within thresholds, decreases towards 0.0 as spread increases
                    straightness_from_link_spread = max(0.0, 1.0 - self.link_spread_penalty_factor * total_spread_excess)
                else:
                    if not (pos_link2 and isinstance(pos_link2, (list, tuple)) and len(pos_link2) == 3):
                        print(f"Warning: CoppeliaMountainEnv (_calculate_downward_straightness_reward) - Could not get valid world position for {self.link2_name}. Received: {pos_link2}.")
                    if not (pos_link3 and isinstance(pos_link3, (list, tuple)) and len(pos_link3) == 3):
                            print(f"Warning: CoppeliaMountainEnv (_calculate_downward_straightness_reward) - Could not get valid world position for {self.link3_name}. Received: {pos_link3}.")
                    # straightness_from_link_spread remains 1.0 (no penalty)
            elif self.link2_handle is None or self.link3_handle is None:
                    print(f"Warning: CoppeliaMountainEnv (_calculate_downward_straightness_reward) - Handles for {self.link2_name} or {self.link3_name} not available. Skipping spread calculation.")
                # straightness_from_link_spread remains 1.0

            # Combine the two straightness measures by multiplication
            combined_straightness = straightness_from_tilt * straightness_from_link_spread
            reward = self.downward_straightness_reward_scale * self.angle_max_reward * combined_straightness
            
        return reward


    def _calculate_landing_reward(self, orientation_euler_rad: np.ndarray, current_height: float, 
                                  current_vz: float, previous_vz: float) -> float:
        """Calculates rewards/penalties associated with landing."""
        landing_reward = 0.0
        effective_ground_contact_height = self.fall_height_threshold + self.land_detection_height_offset
        
        has_landed_this_step = (previous_vz < self.land_detection_falling_vz_thresh and
                                current_vz >= self.land_detection_landed_vz_thresh and
                                current_height < effective_ground_contact_height)

        if has_landed_this_step:
            impact_penalty = -self.land_impact_vz_penalty_scale * abs(previous_vz)
            landing_reward += impact_penalty
            
            impact_tilt_rad = np.sqrt(orientation_euler_rad[0]**2 + orientation_euler_rad[1]**2)
            if impact_tilt_rad < self.land_impact_max_tilt_rad:
                landing_reward += self.land_impact_upright_bonus
                
        return landing_reward

    def _calculate_jump_count_event_reward(self, current_height: float) -> float:
        """
        Calculates reward for crossing the jump height threshold upwards.
        Returns a fixed reward value if a new jump event is detected.
        """
        reward_for_jump_event = 0.0
        is_currently_above = current_height > self.jump_count_height_threshold

        if is_currently_above and not self._was_above_jump_count_threshold:
            reward_for_jump_event = self.jump_count_reward_value
        
        self._was_above_jump_count_threshold = is_currently_above
        return reward_for_jump_event

    def _check_fall_termination_and_penalty(self, current_height: float) -> tuple[bool, float]:
        """Checks if the robot has fallen (hit the ground) and returns termination status and penalty."""
        terminated_by_fall = False
        fall_penalty = 0.0
        
        if current_height < self.fall_height_threshold:
            terminated_by_fall = True
            # fall_penalty = -200.0 
        return terminated_by_fall, fall_penalty

    # --- Core Gym Methods ---
    def _get_observation(self):
        rgb, lidar = self.sim_iface.get_sensor_data()
        j_pos, j_vel = self.sim_iface._get_joint_states()
        base_lin_vel_sim, base_ang_vel_sim = self.sim_iface.get_base_imu_data() 
        obs = {}

        if self.sensor_config['type'] == 'camera':
            res = self.sensor_config.get('resolution', (64,64))
            if self.mode == 'joints_only': img = np.zeros((res[1], res[0], 3), dtype=np.uint8)
            elif rgb is None: img = np.zeros((res[1], res[0], 3), dtype=np.uint8)
            elif rgb.shape[:2] != (res[1], res[0]): img = cv2.resize(rgb, res, interpolation=cv2.INTER_AREA)
            else: img = rgb
            obs["image"] = img
        elif self.sensor_config['type'] == 'lidar':
            pts = self.sensor_config.get('max_points',500)
            if lidar is None or lidar.shape[0]==0: l_pts = np.zeros((pts,3),dtype=np.float32)
            else:
                num_avail = lidar.shape[0]
                if num_avail>=pts: l_pts = lidar[:pts,:]
                else: l_pts = np.vstack((lidar, np.zeros((pts-num_avail,3),dtype=np.float32)))
            obs["lidar_points"] = l_pts
        
        aliases = ['/Joint_0','/Joint_1','/Joint_2','/Joint_3','/joint_rw0','/joint_rw1','/joint_rwz']
        pos_arr = np.array([j_pos.get(k,0.0) for k in aliases],dtype=np.float32)
        vel_arr = np.array([j_vel.get(k,0.0) for k in aliases],dtype=np.float32)
        obs["joint_states"] = np.concatenate([pos_arr, vel_arr])
        obs["imu_data"] = np.concatenate([base_lin_vel_sim, base_ang_vel_sim]) 
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        params = self.default_scene_params
        if options and "scene_params" in options: params = options["scene_params"]
        elif self.current_scene_params is not self.default_scene_params: params = self.current_scene_params
        
        if self.mode == 'joints_only': self.sim_iface.reset_environment(sequential=True)
        else: self.sim_iface.reset_environment(**params)

        self.current_step_count = 0
        self.previous_base_lin_vel_z = 0.0 
        self._was_above_jump_count_threshold = False

        obs = self._get_observation()
        if "imu_data" in obs: 
            self.previous_base_lin_vel_z = obs["imu_data"][2] 

        if self.render_mode == "human": self.render()
        return obs, {}

    def step(self, action):
        self.sim_iface.control(action)
        self.current_step_count += 1
        obs = self._get_observation()
        
        total_reward = 0.0
        terminated = False
        truncated = False
        
        base_lin_vel = obs["imu_data"][:3]
        base_ang_vel = obs["imu_data"][3:]
        current_vz = base_lin_vel[2] 

        current_height = 0.0
        orientation_euler_rad = np.array([0.0, 0.0, 0.0], dtype=np.float32) 

        if self.sim_iface.robot_base:
            pos = self.sim_iface.sim.getObjectPosition(self.sim_iface.robot_base, -1) 
            if pos and isinstance(pos, (list, tuple)) and len(pos) == 3:
                current_height = float(pos[2])
            else:
                print(f"Warning: CoppeliaMountainEnv (step) - Could not get valid position for robot_base handle '{self.sim_iface.robot_base}'. Received: {pos}. Terminating.")
                terminated = True
                total_reward = -500 
                return obs, total_reward, terminated, truncated, {}


            sim_orient_euler = self.sim_iface.sim.getObjectOrientation(self.sim_iface.robot_base, -1) 
            if sim_orient_euler and isinstance(sim_orient_euler, (list, tuple)) and len(sim_orient_euler) == 3:
                orientation_euler_rad = np.array(sim_orient_euler, dtype=np.float32)
            else:
                print(f"Warning: CoppeliaMountainEnv (step) - Could not get valid orientation for robot_base handle '{self.sim_iface.robot_base}'. Received: {sim_orient_euler}. Using default orientation.")
            
            # reward_spin = self._calculate_spin_penalty(base_ang_vel)
            # reward_upward = self._calculate_upward_direction_reward(base_lin_vel)
            # reward_downward_straight = self._calculate_downward_straightness_reward(orientation_euler_rad, current_vz) 
            # reward_landing = self._calculate_landing_reward(orientation_euler_rad, current_height, current_vz, self.previous_base_lin_vel_z)
            reward_jump_count_event = self._calculate_jump_count_event_reward(current_height)
            
            # total_reward = (15 * reward_spin + 
            #                 50 * reward_upward + 
            #                 20 * reward_downward_straight + 
            #                 30 * reward_landing)
            total_reward = self.jump_count_reward_weight * reward_jump_count_event
            
            terminated_by_fall, fall_penalty = self._check_fall_termination_and_penalty(current_height)
            if terminated_by_fall:
                terminated = True
                total_reward = fall_penalty 
        else:
            print("Warning: CoppeliaMountainEnv (step) - self.sim_iface.robot_base is not valid. Terminating episode.")
            terminated = True
            total_reward = -500 

        self.previous_base_lin_vel_z = current_vz

        if not terminated and self.current_step_count >= self.max_episode_steps:
            truncated = True
        
        if self.render_mode == "human":
            self.render()
            
        return obs, total_reward, terminated, truncated, {}

    def render(self):
        if self.render_mode in ["human","rgb_array"] and self.sensor_config['type']=='camera':
            img_obs = self._get_observation() 
            img = img_obs.get("image")
            if img is not None:
                display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if self.render_mode=="human":
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.imshow(self.window_name, display_img)
                        cv2.waitKey(1)
                    else: 
                        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(self.window_name, display_img)
                        cv2.waitKey(1)
                else: 
                    return display_img
        return None

    def close(self):
        self.sim_iface.close()
        if self.render_mode=="human" and self.sensor_config['type']=='camera':
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)