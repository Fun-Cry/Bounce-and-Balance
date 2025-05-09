# env/simulation_copp.py
import numpy as np
import random
import time
import struct
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Use relative imports as this file is part of the 'env' package
from . import actuator, actuator_param # Assuming actuator.py/actuator_param.py are in 'env/'
from .scene_elements import generate_random_mountain

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=1e-3, q_cal=None, exclusion_radius=0.5):
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
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)
        self.scenery_handles = []

    def _get_joint_states(self):
        pos, vel = {}, {}
        for alias, h in self.joint_handles.items():
            try: pos[alias] = self.sim.getJointPosition(h)
            except: pos[alias] = 0.0
            try: vel[alias] = self.sim.getJointVelocity(h)
            except: vel[alias] = 0.0
        return pos, vel

    def control(self, u):
        u = -u; pos, vel = self._get_joint_states()
        if '/Joint_0' not in pos or '/Joint_1' not in pos: return
        q0,q2 = pos['/Joint_0']+self.q_cal[0], pos['/Joint_1']+self.q_cal[1]
        dq0,dq2 = vel['/Joint_0'], vel['/Joint_1']
        ts0,ts1 = self.spring_fn(q0=q0,q2=q2)
        taus = [0.0]*5
        taus[0],_,_ = self.actuator_q0.actuate(i=u[0], q_dot=dq0)
        taus[1],_,_ = self.actuator_q2.actuate(i=u[1], q_dot=dq2)
        taus[2],_,_ = self.actuator_rw1.actuate(i=u[2], q_dot=vel.get('/joint_rw0',0.0))
        taus[3],_,_ = self.actuator_rw2.actuate(i=u[3], q_dot=vel.get('/joint_rw1',0.0))
        taus[4],_,_ = self.actuator_rwz.actuate(i=u[4], q_dot=vel.get('/joint_rwz',0.0))
        taus[0] += ts0; taus[1] += ts1
        mapping = {'/Joint_0':taus[0],'/Joint_1':taus[1],'/joint_rw0':taus[2],'/joint_rw1':taus[3],'/joint_rwz':taus[4]}
        for alias, torque in mapping.items():
            h = self.joint_handles.get(alias)
            if not h: continue
            vcmd = 1000.0 if torque > 0 else (-1000.0 if torque < 0 else 0.0)
            self.sim.setJointTargetVelocity(h, vcmd)
            self.sim.setJointTargetForce(h, float(abs(torque)))
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

    def reset_environment(self, sequential=False, **kwargs):
        if self.robot_base:
            self.sim.setObjectPosition(self.robot_base, -1, self.initial_base_pos)
            self.sim.setObjectOrientation(self.robot_base, -1, self.initial_base_ori)
        if not sequential: self.spawn_scenery(**kwargs)

    def close(self):
        try: self.clear_scenery(); self.sim.stopSimulation()
        except: pass
        cv2.destroyAllWindows(); print('CoppeliaSimZMQInterface closed.')