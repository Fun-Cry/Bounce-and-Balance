# env/simulation_copp.py
import numpy as np
import random
import time
import struct
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Use relative imports as this file is part of the 'env' package
from . import actuator, actuator_param # Restored actuator imports
from .scene_elements import generate_random_mountain

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=5e-2, q_cal=None, exclusion_radius=0.5,
                 direct_max_torques=None): # Added direct_max_torques
        if q_cal is None: q_cal = np.zeros(2)
        self.dt = dt
        self.q_cal = np.array(q_cal)
        self.exclusion_radius = exclusion_radius
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        try:
            self.robot_base = self.sim.getObject('/base_link_respondable')
            self.initial_base_pos = self.sim.getObjectPosition(self.robot_base, -1)
            self.initial_base_ori = self.sim.getObjectOrientation(self.robot_base, -1)
        except Exception:
            self.robot_base = None; self.initial_base_pos = [0,0,0]; self.initial_base_ori = [0,0,0]
            print('⚠️ Could not get /base_link_respondable. Robot height reward & fall detection might be affected.')
        try:
            self.vision_sensor_handle = self.sim.getObject('/base_link_respondable/visionSensor')
        except: self.vision_sensor_handle = None; print('⚠️ Vision sensor /base_link_respondable/visionSensor not found.')
        try:
            self.lidar_handle = self.sim.getObject('/base_link_respondable/VelodyneVPL16')
            self.lidar_script = self.sim.getScript(self.sim.scripttype_childscript, self.lidar_handle)
        except: self.lidar_handle = None; self.lidar_script = None; print('⚠️ LiDAR /base_link_respondable/VelodyneVPL16 or script not found.')

        self.sim.setArrayParam(self.sim.arrayparam_gravity, [0, 0, -9.81])
        if hasattr(self.sim, 'setStepping'): self.sim.setStepping(True)

        if joint_aliases is None:
            joint_aliases = ['/Joint_0','/Joint_1','/Joint_2','/Joint_3','/joint_rw0','/joint_rw1','/joint_rwz']
        self.joint_handles = {}
        for alias in joint_aliases:
            try: self.joint_handles[alias] = self.sim.getObject(alias)
            except Exception as e: print(f'Warning: could not get handle for {alias}: {e}')

        self.sim.startSimulation()
        self.spring_fn = spring.fn_spring
        
        # Original Actuator initializations for control_old
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10) 
        self.actuator_rw1= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)

        # Initialization for new 'control' method
        if direct_max_torques is None:
            # Default max torques for direct control: [J0, J1, RW0, RW1, RWZ]
            self.direct_max_torques = np.array([7.143, 7.143, 11.24, 11.24, 4.71], dtype=np.float32)
        else:
            self.direct_max_torques = np.array(direct_max_torques, dtype=np.float32)
        
        if len(self.direct_max_torques) != 5:
            raise ValueError("direct_max_torques must be a list or array of 5 values.")

        self.scenery_handles = []
        self.controlled_joint_aliases = ['/Joint_0', '/Joint_1', '/joint_rw0', '/joint_rw1', '/joint_rwz']


    def _get_joint_states(self):
        pos, vel = {}, {}
        for alias, h in self.joint_handles.items():
            pos[alias] = self.sim.getJointPosition(h)
            vel[alias] = self.sim.getJointVelocity(h)
        return pos, vel

    def get_base_imu_data(self):
        """Gets linear and angular velocity of the robot base."""
        if self.robot_base:
            try:
                linear_vel, angular_vel = self.sim.getObjectVelocity(self.robot_base)
                return np.array(linear_vel, dtype=np.float32), np.array(angular_vel, dtype=np.float32)
            except Exception as e:
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def control(self, action_from_rl): # Original control method using actuators
        processed_action = -np.asarray(action_from_rl, dtype=np.float32)
        pos, vel = self._get_joint_states()

        if '/Joint_0' not in pos or '/Joint_1' not in pos:
            return

        q0_spring = pos.get('/Joint_0', 0.0) + self.q_cal[0]
        q1_spring_for_q2_param = pos.get('/Joint_1', 0.0) + self.q_cal[1]
        ts0, ts1 = self.spring_fn(q0=q0_spring, q2=q1_spring_for_q2_param)

        taus_from_actuators = [0.0] * 5

        target_current_q0 = processed_action[0] * self.actuator_q0.i_max
        dq0 = vel.get('/Joint_0', 0.0)
        taus_from_actuators[0], _, _ = self.actuator_q0.actuate(i=target_current_q0, q_dot=dq0)

        target_current_j1 = processed_action[1] * self.actuator_q2.i_max
        dq1 = vel.get('/Joint_1', 0.0)
        taus_from_actuators[1], _, _ = self.actuator_q2.actuate(i=target_current_j1, q_dot=dq1)

        target_current_rw0 = processed_action[2] * self.actuator_rw1.i_max
        dqrw0 = vel.get('/joint_rw0', 0.0)
        taus_from_actuators[2], _, _ = self.actuator_rw1.actuate(i=target_current_rw0, q_dot=dqrw0)

        target_current_rw1 = processed_action[3] * self.actuator_rw2.i_max
        dqrw1 = vel.get('/joint_rw1', 0.0)
        taus_from_actuators[3], _, _ = self.actuator_rw2.actuate(i=target_current_rw1, q_dot=dqrw1)

        target_current_rwz = processed_action[4] * self.actuator_rwz.i_max
        dqrwz = vel.get('/joint_rwz', 0.0)
        taus_from_actuators[4], _, _ = self.actuator_rwz.actuate(i=target_current_rwz, q_dot=dqrwz)
        
        final_torques_to_apply = list(taus_from_actuators) 
        final_torques_to_apply[0] += ts0 
        final_torques_to_apply[1] += ts1 

        joint_torque_mapping = {
            self.controlled_joint_aliases[0]: final_torques_to_apply[0], # /Joint_0
            self.controlled_joint_aliases[1]: final_torques_to_apply[1], # /Joint_1
            self.controlled_joint_aliases[2]: final_torques_to_apply[2], # /joint_rw0
            self.controlled_joint_aliases[3]: final_torques_to_apply[3], # /joint_rw1
            self.controlled_joint_aliases[4]: final_torques_to_apply[4]  # /joint_rwz
        }

        for alias, torque_value in joint_torque_mapping.items():
            joint_handle = self.joint_handles.get(alias)
            if not joint_handle:
                continue
            
            target_velocity_for_torque_mode = 1000.0 
            if torque_value == 0.0:
                self.sim.setJointTargetVelocity(joint_handle, 0.0)
                self.sim.setJointTargetForce(joint_handle, 0.0) 
            elif torque_value > 0:
                self.sim.setJointTargetVelocity(joint_handle, target_velocity_for_torque_mode)
                self.sim.setJointTargetForce(joint_handle, float(abs(torque_value)))
            else: # torque_value < 0
                self.sim.setJointTargetVelocity(joint_handle, -target_velocity_for_torque_mode)
                self.sim.setJointTargetForce(joint_handle, float(abs(torque_value)))
        
        self.sim.step()

    def control_raw(self, action_from_rl): # New control method for direct torque control
        processed_action = -np.asarray(action_from_rl, dtype=np.float32)
        pos, _ = self._get_joint_states() # vel is not directly used here for torque calculation

        if '/Joint_0' not in pos or '/Joint_1' not in pos:
            return

        q0_spring = pos.get('/Joint_0', 0.0) + self.q_cal[0]
        q1_spring_for_q2_param = pos.get('/Joint_1', 0.0) + self.q_cal[1]
        ts0, ts1 = self.spring_fn(q0=q0_spring, q2=q1_spring_for_q2_param)

        taus_from_actions = [0.0] * 5

        # Directly apply scaled actions as torques
        taus_from_actions[0] = processed_action[0] * self.direct_max_torques[0]
        taus_from_actions[1] = processed_action[1] * self.direct_max_torques[1]
        taus_from_actions[2] = processed_action[2] * self.direct_max_torques[2]
        taus_from_actions[3] = processed_action[3] * self.direct_max_torques[3]
        taus_from_actions[4] = processed_action[4] * self.direct_max_torques[4]
        
        final_torques_to_apply = list(taus_from_actions) 
        final_torques_to_apply[0] += ts0 
        final_torques_to_apply[1] += ts1 

        joint_torque_mapping = {
            self.controlled_joint_aliases[0]: final_torques_to_apply[0], # /Joint_0
            self.controlled_joint_aliases[1]: final_torques_to_apply[1], # /Joint_1
            self.controlled_joint_aliases[2]: final_torques_to_apply[2], # /joint_rw0
            self.controlled_joint_aliases[3]: final_torques_to_apply[3], # /joint_rw1
            self.controlled_joint_aliases[4]: final_torques_to_apply[4]  # /joint_rwz
        }
        # print(joint_torque_mapping)

        for alias, torque_value in joint_torque_mapping.items():
            joint_handle = self.joint_handles.get(alias)
            if not joint_handle:
                continue
            
            target_velocity_for_torque_mode = 1000.0 
            if torque_value == 0.0:
                self.sim.setJointTargetVelocity(joint_handle, 0.0)
                self.sim.setJointTargetForce(joint_handle, 0.0) 
            elif torque_value > 0:
                self.sim.setJointTargetVelocity(joint_handle, target_velocity_for_torque_mode)
                self.sim.setJointTargetForce(joint_handle, float(abs(torque_value)))
            else: # torque_value < 0
                self.sim.setJointTargetVelocity(joint_handle, -target_velocity_for_torque_mode)
                self.sim.setJointTargetForce(joint_handle, float(abs(torque_value)))
        
        self.sim.step()

    def get_sensor_data(self, show=False):
        rgb, lidar = None, None
        if self.vision_sensor_handle:
            try:
                rx,ry=self.sim.getVisionSensorResolution(self.vision_sensor_handle)
                buf,_,_=self.sim.getVisionSensorCharImage(self.vision_sensor_handle)
                rgb=np.frombuffer(buf,np.uint8).reshape((ry,rx,3))[::-1]
                if show and rgb is not None: cv2.imshow('RGB from Sim',rgb); cv2.waitKey(1)
            except: pass
        if self.lidar_handle and self.lidar_script:
            try:
                _,_,_,buf=self.sim.callScriptFunction('getVelodyneBuffer',self.lidar_script,[],[],[],b'')
                if buf: lidar=np.array(struct.unpack('f'*(len(buf)//4),buf),np.float32).reshape(-1,3)
                if show and lidar is not None: print(f'[LiDAR from Sim] {lidar.shape[0]} pts')
            except: pass
        return rgb, lidar

    def clear_scenery(self):
        for h in self.scenery_handles:
            try: self.sim.removeObject(h)
            except: pass
        self.scenery_handles = []

    def spawn_scenery(self, n_items=3, shape_options=8, mountain_target_total_height=1.0,
                        mountain_max_cylinder_height=0.4, mountain_min_cylinder_height=0.05,
                        mountain_peak_radius=0.1, mountain_base_radius_factor_range=(4.0, 7.0),
                        mountain_area_bounds_x=None, mountain_area_bounds_y=None, **kwargs):
        self.clear_scenery()
        for _ in range(n_items):
            factor = random.uniform(mountain_base_radius_factor_range[0], mountain_base_radius_factor_range[1])
            args={'sim':self.sim,'target_total_height':mountain_target_total_height,
                    'max_cylinder_height':mountain_max_cylinder_height,'min_cylinder_height':mountain_min_cylinder_height,
                    'peak_radius':mountain_peak_radius,'base_radius_factor':factor,'shape_options':shape_options}
            if mountain_area_bounds_x: args['area_bounds_x']=mountain_area_bounds_x
            if mountain_area_bounds_y: args['area_bounds_y']=mountain_area_bounds_y
            handles = generate_random_mountain(**args)
            if handles: self.scenery_handles.extend(handles)

    def reset_environment(self, sequential=False, initial_leg_angles=None, **kwargs):
        if self.robot_base:
            self.sim.setObjectPosition(self.robot_base, -1, self.initial_base_pos)
            self.sim.setObjectOrientation(self.robot_base, -1, self.initial_base_ori)


        if initial_leg_angles is None:
            initial_leg_angles = {'/Joint_0': 0.0, '/Joint_1': 0.0, '/Joint_2': 0.0, '/Joint_3': 0.0} 
        
        for alias in self.controlled_joint_aliases:
            joint_handle = self.joint_handles.get(alias)
            if joint_handle:
                if alias in initial_leg_angles:
                    self.sim.setJointPosition(joint_handle, initial_leg_angles[alias])
                elif 'rw' in alias: 
                    self.sim.setJointPosition(joint_handle, 0.0)

                self.sim.setJointTargetVelocity(joint_handle, 0.0)
                self.sim.setJointTargetForce(joint_handle, 0.0) 
                
        if not sequential: 
            # self.spawn_scenery(**kwargs)
            pass

    def close(self):
        try: 
            self.clear_scenery()
            current_sim_state = self.sim.getSimulationState()
            if current_sim_state != self.sim.simulation_stopped:
                self.sim.stopSimulation()
        except Exception as e: 
            print(f"Exception during stopSimulation: {e}")
        finally:
            cv2.destroyAllWindows()
            print('CoppeliaSimZMQInterface resources released.')