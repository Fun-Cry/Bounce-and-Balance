import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.simulation_copp import CoppeliaSimZMQInterface
import cv2

class HopperCoppeliaEnv(gym.Env):
    """
    Gym interface for Hopper in CoppeliaSim via ZMQ Remote API.
    Observations: robot state + visual data (camera or LiDAR) detected at runtime.
    Actions: 5-element torque vector.
    Reward: base height.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 spring_model,
                 max_lidar_points: int = 1000,
                 max_steps: int = 1000,
                 env_kwargs: dict = None):
        super().__init__()
        # Simulation interface
        self.sim = CoppeliaSimZMQInterface(spring=spring_model)
        self.max_lidar_points = max_lidar_points
        self.max_steps = max_steps
        self.step_count = 0
        self.env_kwargs = env_kwargs or {
            'n_towers': 6,
            'area_radius': 2.0,
            'radius_range': (0.2,0.5),
            'height_range': (0.1,0.4),
            'shape_options': 9
        }

        # Action: 5 torques in [-1,1]
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        # Placeholder observation_space; will be set on reset
        self.observation_space = None

    def reset(self):
        # Reset sim and robot
        self.sim.reset_environment(**self.env_kwargs)
        self.step_count = 0

        # Peek sensor to configure observation_space
        rgb, lidar = self.sim.get_sensor_data(show=False)
        # Joint-state space
        state_low  = np.full((5,), -np.inf, np.float32)
        state_high = np.full((5,),  np.inf, np.float32)
        obs_spaces = {'state': spaces.Box(state_low, state_high, dtype=np.float32)}

        # Visual space depends on which sensor is active
        if rgb is not None:
            # camera: uint8 image
            h, w, c = rgb.shape
            obs_spaces['vision'] = spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
        elif lidar is not None:
            # LiDAR: (N,3) float32 padded/truncated
            obs_spaces['vision'] = spaces.Box(-np.inf, np.inf,
                                              (self.max_lidar_points, 3),
                                              dtype=np.float32)
        else:
            # no sensor
            obs_spaces['vision'] = spaces.Box(-np.inf, np.inf, (0,0), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_spaces)
        return self._get_obs(current_rgb=rgb, current_lidar=lidar)

    def step(self, action):
        a = np.clip(action, self.action_space.low, self.action_space.high)
        self.sim.control(a)
        obs = self._get_obs()
        # Reward = base height
        state = obs['state']
        reward = state[4]
        self.step_count += 1
        done = self.step_count >= self.max_steps or state[4] < 0.1
        if state[4] < 0.1:
            reward -= 1.0
        return obs, reward, done, {}

    def _get_obs(self, current_rgb=None, current_lidar=None):
        # Joint state
        pos, vel = self.sim._get_joint_states()
        q0 = pos.get('/Joint_0', 0.0)
        q1 = pos.get('/Joint_1', 0.0)
        dq0 = vel.get('/Joint_0', 0.0)
        dq1 = vel.get('/Joint_1', 0.0)
        base_z = 0.0
        if hasattr(self.sim, 'robot_base') and self.sim.robot_base:
            base_z = self.sim.sim.getObjectPosition(self.sim.robot_base, -1)[2]
        state = np.array([q0, q1, dq0, dq1, base_z], np.float32)

        # Visual data
        if current_rgb is None and current_lidar is None:
            rgb, lidar = self.sim.get_sensor_data(show=False)
        else:
            rgb, lidar = current_rgb, current_lidar

        if rgb is not None:
            vision = rgb
        elif lidar is not None:
            pts = lidar[:self.max_lidar_points]
            if pts.shape[0] < self.max_lidar_points:
                pad = np.zeros((self.max_lidar_points - pts.shape[0], 3), np.float32)
                pts = np.vstack([pts, pad])
            vision = pts
        else:
            vision = np.zeros(self.observation_space['vision'].shape,
                               dtype=self.observation_space['vision'].dtype)

        return {'state': state, 'vision': vision}

    def render(self, mode='human'):
        # simulation windows already show sensors
        pass

    def close(self):
        self.sim.close()
        super().close()
