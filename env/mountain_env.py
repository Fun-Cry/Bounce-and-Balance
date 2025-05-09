# env/mountain_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from .simulation_copp import CoppeliaSimZMQInterface

class DummySpring:
    def fn_spring(self, q0, q2): return 0.0, 0.0

class CoppeliaMountainEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, scene_params=None, sensor_config=None, max_episode_steps=1000):
        super().__init__()
        self.sim_iface = CoppeliaSimZMQInterface(spring=DummySpring())
        self.render_mode = render_mode
        self.sensor_config = sensor_config if sensor_config else {'type': 'camera', 'resolution': (64, 64)}
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
        self.fall_height_threshold = 0.1
        self.fall_angle_threshold = np.deg2rad(60)

    def _get_observation(self):
        rgb,lidar = self.sim_iface.get_sensor_data()
        j_pos,j_vel = self.sim_iface._get_joint_states()
        obs = {}
        if self.sensor_config['type'] == 'camera':
            res = self.sensor_config.get('resolution',(64,64))
            if rgb is None: img = np.zeros((res[1],res[0],3),dtype=np.uint8)
            elif rgb.shape[:2]!=(res[1],res[0]): img = cv2.resize(rgb,res,interpolation=cv2.INTER_AREA)
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
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        params = self.default_scene_params
        if options and "scene_params" in options: params = options["scene_params"]
        elif self.current_scene_params is not self.default_scene_params: params = self.current_scene_params
        self.sim_iface.reset_environment(**params)
        self.current_step_count = 0
        obs = self._get_observation()
        if self.render_mode == "human": self.render()
        return obs, {}

    def step(self, action):
        self.sim_iface.control(action); self.current_step_count+=1
        obs = self._get_observation()
        reward,terminated,truncated = 0.0,False,False
        
        if self.sim_iface.robot_base:
            try:
                pos = self.sim_iface.sim.getObjectPosition(self.sim_iface.robot_base,-1)
                height = float(pos[2]); reward = height
                orient = self.sim_iface.sim.getObjectOrientation(self.sim_iface.robot_base,-1)
                tilted = abs(orient[1])>self.fall_angle_threshold or abs(orient[2])>self.fall_angle_threshold
                low = height < self.fall_height_threshold
                if tilted or low: terminated=True; reward=-200.0
            except: pass
        
        if not terminated and self.current_step_count>=self.max_episode_steps: truncated=True
        if self.render_mode=="human": self.render()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode in ["human","rgb_array"] and self.sensor_config['type']=='camera':
            img = self._get_observation().get("image")
            if img is not None:
                if self.render_mode=="human": cv2.imshow(self.window_name,cv2.cvtColor(img,cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
                else: return img
        return None

    def close(self):
        self.sim_iface.close()
        if self.render_mode=="human" and self.sensor_config['type']=='camera':
            if cv2.getWindowProperty(self.window_name,cv2.WND_PROP_VISIBLE)>=1: cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)