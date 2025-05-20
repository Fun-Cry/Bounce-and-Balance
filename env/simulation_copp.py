# env/simulation_copp.py 
# (Make sure this is the file with CoppeliaSimZMQInterface)
import numpy as np
import random
import time
import struct
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from . import actuator, actuator_param 
from .scene_elements import generate_random_mountain

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=5e-2, q_cal=None, exclusion_radius=0.5):
        if q_cal is None: q_cal = np.zeros(2)
        self.dt = dt
        self.q_cal = np.array(q_cal)
        self.exclusion_radius = exclusion_radius
        self.client = RemoteAPIClient() 
        self.sim = self.client.getObject('sim') 

        self.SCRIPT_CALL_SUCCESS = 1 
        if hasattr(self.sim, 'script_call_success'):
            self.SCRIPT_CALL_SUCCESS = self.sim.script_call_success
        elif hasattr(self.sim, 'simx_return_ok'): 
             self.SCRIPT_CALL_SUCCESS = self.sim.simx_return_ok

        self.robot_base = -1 
        self.initial_base_pos = [0,0,0.3] 
        self.initial_base_ori = [0,0,0]
        try:
            self.robot_base = self.sim.getObject('/base_link_respondable')
            if self.robot_base != -1:
                scene_initial_pos = self.sim.getObjectPosition(self.robot_base, -1)
                scene_initial_ori = self.sim.getObjectOrientation(self.robot_base, -1)
                if scene_initial_pos: self.initial_base_pos = scene_initial_pos
                if scene_initial_ori: self.initial_base_ori = scene_initial_ori
                print(f"INFO: Robot base '/base_link_respondable' found. Handle: {self.robot_base}. Initial scene pose: P={self.initial_base_pos}, O={self.initial_base_ori}")
            else:
                print("WARNING: Robot base '/base_link_respondable' not found. Using default initial pose.")
        except Exception as e:
            print(f"WARNING: Error getting /base_link_respondable: {e}. Using default initial pose.")
        
        self.vision_sensor_handle = -1
        try:
            self.vision_sensor_handle = self.sim.getObject('/base_link_respondable/visionSensor') # Ensure this path is correct in your scene
            if self.vision_sensor_handle == -1:
                 print('WARNING: Vision sensor /base_link_respondable/visionSensor not found.')
        except Exception as e: print(f'WARNING: Error getting vision sensor: {e}.')
        
        self.lidar_handle = -1
        self.lidar_script_handle = -1 
        try:
            self.lidar_handle = self.sim.getObject('/base_link_respondable/VelodyneVPL16')
            if self.lidar_handle != -1:
                 self.lidar_script_handle = self.sim.getScript(self.sim.scripttype_childscript, self.lidar_handle)
                 if self.lidar_script_handle == -1:
                     print('WARNING: LiDAR script for /base_link_respondable/VelodyneVPL16 not found.')
            # else: # This warning is expected if not using LiDAR
            #     print('INFO: LiDAR /base_link_respondable/VelodyneVPL16 not found (this is OK if using camera).')
        except Exception as e: print(f'WARNING: Error getting LiDAR: {e}.') # This warning is expected

        self.sim.setArrayParam(self.sim.arrayparam_gravity, [0, 0, -9.81])
        self.sim.setStepping(True) 

        self.all_joint_aliases = ['/Joint_0','/Joint_1','/Joint_2','/Joint_3','/joint_rw0','/joint_rw1','/joint_rwz']
        if joint_aliases:
             self.all_joint_aliases = list(set(self.all_joint_aliases + joint_aliases))

        self.joint_handles = {}
        for alias in self.all_joint_aliases:
            try: 
                handle = self.sim.getObject(alias)
                if handle != -1:
                    self.joint_handles[alias] = handle
            except Exception as e: 
                print(f'Warning: Could not get handle for {alias}: {e}')
        
        print(f"INFO: Fetched handles for joints: {list(self.joint_handles.keys())}")

        current_sim_state = self.sim.getSimulationState()
        if current_sim_state == self.sim.simulation_stopped:
            self.sim.startSimulation()
            print("INFO: Simulation started by CoppeliaSimZMQInterface.")
        else:
            print("INFO: Simulation was already running.")
        
        self.spring_fn = spring.fn_spring
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q1 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw0= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw1= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)
        
        self.scenery_handles = []

        self.controlled_joint_aliases_ordered = [
            '/Joint_0', '/Joint_1', '/joint_rw0', '/joint_rw1', '/joint_rwz'
        ] 
        self.actuators_ordered = [
            self.actuator_q0, self.actuator_q1, self.actuator_rw0,
            self.actuator_rw1, self.actuator_rwz
        ]

    def _get_joint_states(self):
        pos, vel = {}, {}
        for alias, h in self.joint_handles.items():
            if h == -1: continue
            try:
                pos[alias] = self.sim.getJointPosition(h)
                vel[alias] = self.sim.getJointVelocity(h)
            except Exception as e:
                pos[alias] = 0.0
                vel[alias] = 0.0
        return pos, vel

    def get_base_imu_data(self):
        if self.robot_base != -1:
            try:
                linear_vel, angular_vel = self.sim.getObjectVelocity(self.robot_base)
                return np.array(linear_vel, dtype=np.float32), np.array(angular_vel, dtype=np.float32)
            except Exception as e:
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def control(self, action_from_controller): 
        processed_action = -np.asarray(action_from_controller, dtype=np.float32) 
        current_joint_pos, current_joint_vel = self._get_joint_states()

        q0_spring_val = current_joint_pos.get('/Joint_0', 0.0) + self.q_cal[0]
        q1_spring_val = current_joint_pos.get('/Joint_1', 0.0) + self.q_cal[1]
        ts0, ts1 = self.spring_fn(q0=q0_spring_val, q2=q1_spring_val)

        final_torques_to_apply = [0.0] * len(self.controlled_joint_aliases_ordered)

        # !!!!! EMERGENCY CHANGE: SCALE DOWN REACTION WHEEL ACTIONS !!!!!
        reaction_wheel_action_scale = 0.05 # Try 5% of original authority. You can tune this (e.g., 0.01 to 0.1)

        for i, alias in enumerate(self.controlled_joint_aliases_ordered):
            # Get the original action value for this joint from the policy
            current_action_value_from_policy = processed_action[i]
            
            # If this is a reaction wheel joint, drastically scale down its commanded action value
            if alias in ['/joint_rw0', '/joint_rw1', '/joint_rwz']:
                # Uncomment for debugging if needed:
                # print(f"Original RW action for {alias}: {current_action_value_from_policy:.4f}, Scaled: {current_action_value_from_policy * reaction_wheel_action_scale:.4f}")
                current_action_value_from_policy *= reaction_wheel_action_scale
            
            # Calculate target current using the (potentially scaled) action value
            target_current = current_action_value_from_policy * self.actuators_ordered[i].i_max
            joint_velocity = current_joint_vel.get(alias, 0.0)
            
            tau_actuator, _, _ = self.actuators_ordered[i].actuate(i=target_current, q_dot=joint_velocity)
            final_torques_to_apply[i] = tau_actuator

        # Add spring torques (if any) to the appropriate leg joints
        final_torques_to_apply[0] += ts0 
        final_torques_to_apply[1] += ts1 

        # Apply the final torques to the joints in the simulation
        for i, alias in enumerate(self.controlled_joint_aliases_ordered):
            joint_handle = self.joint_handles.get(alias)
            if not joint_handle or joint_handle == -1:
                continue
            
            torque_value = final_torques_to_apply[i]
            target_velocity_for_torque_mode = 10000.0 # High target velocity for torque control mode
            
            try:
                if torque_value == 0.0:
                    self.sim.setJointTargetVelocity(joint_handle, 0.0)
                    # Set a very small force to allow the joint to be "loose" if no torque is commanded
                    self.sim.setJointTargetForce(joint_handle, 0.001) 
                elif torque_value > 0:
                    self.sim.setJointTargetVelocity(joint_handle, target_velocity_for_torque_mode)
                    self.sim.setJointTargetForce(joint_handle, float(abs(torque_value)))
                else: # torque_value < 0
                    self.sim.setJointTargetVelocity(joint_handle, -target_velocity_for_torque_mode)
                    self.sim.setJointTargetForce(joint_handle, float(abs(torque_value)))
            except Exception as e:
                print(f"Error setting joint torque for {alias} (handle {joint_handle}): {e}")
        
        self.sim.step()

    def get_sensor_data(self, show=False):
        rgb, lidar = None, None
        if self.vision_sensor_handle != -1:
            try:
                rx,ry=self.sim.getVisionSensorResolution(self.vision_sensor_handle)
                buf,_,_=self.sim.getVisionSensorCharImage(self.vision_sensor_handle) 
                if buf: rgb=np.frombuffer(buf,np.uint8).reshape((ry,rx,3))[::-1] 
                if show and rgb is not None: 
                    cv2.imshow('RGB from Sim', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            except Exception as e: 
                pass 
        if self.lidar_handle != -1 and self.lidar_script_handle != -1:
            try:
                ret_code, _, _, _, lidar_buffer = self.sim.callScriptFunction(
                    self.lidar_script_handle,      
                    'getVelodyneData_points',      
                    [], [], [], b''                
                )
                if ret_code == self.SCRIPT_CALL_SUCCESS and lidar_buffer:
                     lidar=np.array(struct.unpack('f'*(len(lidar_buffer)//4),lidar_buffer),np.float32).reshape(-1,3)
                elif ret_code != self.SCRIPT_CALL_SUCCESS:
                    pass 
                if show and lidar is not None: print(f'[LiDAR from Sim] {lidar.shape[0]} pts')
            except Exception as e:
                pass
        return rgb, lidar

    def clear_scenery(self):
        for h in self.scenery_handles:
            if h != -1:
                try: self.sim.removeObject(h) 
                except: pass
        self.scenery_handles = []

    def spawn_scenery(self, n_items=3, **kwargs_for_mountain_generation): 
        self.clear_scenery()
        scene_creation_params = {
            'sim': self.sim, 
            'target_total_height': kwargs_for_mountain_generation.get('mountain_target_total_height', 1.0),
            'max_cylinder_height': kwargs_for_mountain_generation.get('mountain_max_cylinder_height', 0.11),
            'min_cylinder_height': kwargs_for_mountain_generation.get('mountain_min_cylinder_height', 0.05),
            'peak_radius': kwargs_for_mountain_generation.get('mountain_peak_radius', 0.3),
            'base_radius_factor': random.uniform(*kwargs_for_mountain_generation.get('mountain_base_radius_factor_range', (3.0, 5.0))),
            'shape_options': kwargs_for_mountain_generation.get('shape_options', 9), 
            'area_bounds_x': kwargs_for_mountain_generation.get('mountain_area_bounds_x', (-2.5, 2.5)),
            'area_bounds_y': kwargs_for_mountain_generation.get('mountain_area_bounds_y', (1.5, 3.5)),
            'exclusion_radius': self.exclusion_radius, 
            'robot_pos': self.initial_base_pos[:2]    
        }
        for _ in range(n_items): 
            handles = generate_random_mountain(**scene_creation_params) 
            if handles: self.scenery_handles.extend(handles)

    def reset_environment(self, sequential=False, initial_joint_angles=None, **kwargs):
        if self.robot_base != -1 :
            try:
                self.sim.setObjectPosition(self.robot_base, -1, self.initial_base_pos)
                self.sim.setObjectOrientation(self.robot_base, -1, self.initial_base_ori)
            except Exception as e:
                print(f"Warning: Could not reset robot base pose: {e}")

        for alias, joint_handle in self.joint_handles.items():
            if joint_handle == -1: continue
            try:
                angle_to_set = 0.0
                if initial_joint_angles and alias in initial_joint_angles:
                     angle_to_set = initial_joint_angles[alias]
                
                self.sim.setJointPosition(joint_handle, angle_to_set)
                self.sim.setJointTargetVelocity(joint_handle, 0.0) 
                self.sim.setJointTargetForce(joint_handle, 0.001) 
            except Exception as e:
                print(f"Warning: Could not reset joint {alias} (handle {joint_handle}): {e}")

        if initial_joint_angles: 
            for alias, angle_val in initial_joint_angles.items():
                joint_handle = self.joint_handles.get(alias)
                if joint_handle and joint_handle != -1:
                    try:
                        self.sim.setJointPosition(joint_handle, angle_val)
                        self.sim.setJointTargetVelocity(joint_handle, 0.0)
                        self.sim.setJointTargetForce(joint_handle, 0.001)
                    except Exception as e:
                        print(f"Warning: Could not set initial angle for joint {alias} (handle {joint_handle}): {e}")

        for act in self.actuators_ordered:
            act.i_smoothed = 0.0

        if not sequential:
            # MODIFIED: Pop 'n_items' from kwargs before passing to spawn_scenery
            # Determine n_items_for_scenery explicitly.
            # The default for 'n_items' in kwargs coming from mountain_env is usually 0 for this setup.
            # Let's respect what mountain_env wants for n_items primarily.
            n_items_from_kwargs = kwargs.pop('n_items', None) # Try to get n_items from kwargs and remove it

            if n_items_from_kwargs is not None:
                n_items_for_scenery = n_items_from_kwargs
            else: # Fallback if not in kwargs (shouldn't happen if mountain_env sends scene_params)
                n_items_for_scenery = 0 if self.initial_base_pos[2] < 0.1 else 3
            
            # Now kwargs will not contain 'n_items', avoiding the TypeError
            self.spawn_scenery(n_items=n_items_for_scenery, **kwargs)
        else:
            self.clear_scenery()
            
    def close(self):
        try:
            self.clear_scenery()
            if self.sim: 
                current_sim_state = self.sim.getSimulationState() 
                if current_sim_state != self.sim.simulation_stopped: 
                    self.sim.stopSimulation() 
                    print("CoppeliaSim simulation stopped by ZMQ interface.")
        except Exception as e:
            print(f"Exception during CoppeliaSim ZMQ interface close: {e}")
        finally:
            cv2.destroyAllWindows()
            print('CoppeliaSimZMQInterface resources released (client will close on exit).')