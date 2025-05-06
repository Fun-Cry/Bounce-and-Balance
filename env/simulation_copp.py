import numpy as np
import time
import struct
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from env import actuator, actuator_param
from env.scene_elements import generate_random_mountain # MODIFIED: Import for mountains
import random

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=1e-3, q_cal=None, exclusion_radius=0.5):
        """
        Initialize simulation interface, actuators, and record initial robot pose.
        :param exclusion_radius: no-tower zone around robot base (meters)
        """
        if q_cal is None:
            q_cal = np.zeros(2)
        self.dt = dt
        self.q_cal = np.array(q_cal)
        self.exclusion_radius = exclusion_radius # Note: Not actively used by mountain generation's placement

        # Remote API client
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # Retrieve robot base link handle and its initial pose
        try:
            self.robot_base = self.sim.getObject('/base_link_respondable')
            self.initial_base_pos = self.sim.getObjectPosition(self.robot_base, -1)
            self.initial_base_ori = self.sim.getObjectOrientation(self.robot_base, -1)
        except Exception:
            self.robot_base = None
            self.initial_base_pos = [0,0,0]
            self.initial_base_ori = [0,0,0]
            print('⚠️ Could not get base_link_respondable. Mountain placement will not actively avoid robot start.')

        # Vision sensor handle
        try:
            self.vision_sensor_handle = self.sim.getObject('/base_link_respondable/visionSensor')
        except:
            self.vision_sensor_handle = None
            print('⚠️ Vision sensor not found.')

        # LiDAR sensor and script
        try:
            self.lidar_handle = self.sim.getObject('/base_link_respondable/VelodyneVPL16')
            self.lidar_script = self.sim.getScript(
                self.sim.scripttype_childscript, self.lidar_handle)
        except:
            self.lidar_handle = None
            self.lidar_script = None
            print('⚠️ LiDAR or child script not found.')

        # Zero gravity and synchronous stepping
        self.sim.setArrayParam(self.sim.arrayparam_gravity, [0, 0, -9.81])
        if hasattr(self.sim, 'setStepping'):
            self.sim.setStepping(True)

        # Joint handles
        if joint_aliases is None:
            joint_aliases = ['/Joint_0','/Joint_1','/Joint_2','/Joint_3',
                             '/joint_rw0','/joint_rw1','/joint_rwz']
        self.joint_handles = {}
        for alias in joint_aliases:
            try:
                self.joint_handles[alias] = self.sim.getObject(alias)
            except Exception as e:
                print(f'Warning: could not get handle for {alias}: {e}')

        # Start simulation and actuators
        self.sim.startSimulation()
        self.spring_fn   = spring.fn_spring
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)

        # Container for scenery handles (was tower_handles)
        self.scenery_handles = [] # Renamed for clarity, was tower_handles

    def _get_joint_states(self):
        pos, vel = {}, {}
        for alias, h in self.joint_handles.items():
            try:
                pos[alias] = self.sim.getJointPosition(h)
            except:
                pos[alias] = 0.0
            try:
                vel[alias] = self.sim.getJointVelocity(h)
            except:
                vel[alias] = 0.0
        return pos, vel

    def control(self, u):
        u = -u
        pos, vel = self._get_joint_states()
        if '/Joint_0' not in pos or '/Joint_1' not in pos:
            return
        q0 = pos['/Joint_0'] + self.q_cal[0]
        q2 = pos['/Joint_1'] + self.q_cal[1]
        dq0, dq2 = vel['/Joint_0'], vel['/Joint_1']
        ts0, ts1 = self.spring_fn(q0=q0, q2=q2)

        taus = [None]*5
        taus[0],_,_ = self.actuator_q0.actuate(i=u[0], q_dot=dq0)
        taus[1],_,_ = self.actuator_q2.actuate(i=u[1], q_dot=dq2)
        taus[2],_,_ = self.actuator_rw1.actuate(i=u[2], q_dot=vel.get('/joint_rw0',0.0))
        taus[3],_,_ = self.actuator_rw2.actuate(i=u[3], q_dot=vel.get('/joint_rw1',0.0))
        taus[4],_,_ = self.actuator_rwz.actuate(i=u[4], q_dot=vel.get('/joint_rwz',0.0))
        taus[0] += ts0; taus[1] += ts1

        # ... (previous lines in control method) ...
        mapping = {'/Joint_0':taus[0],'/Joint_1':taus[1],
                   '/joint_rw0':taus[2],'/joint_rw1':taus[3],'/joint_rwz':taus[4]}
        for alias, torque in mapping.items(): # 'torque' is the variable from the loop
            h = self.joint_handles.get(alias)
            if not h: continue
            vcmd = 1000.0 if torque > 0 else (-1000.0 if torque < 0 else 0.0)
            self.sim.setJointTargetVelocity(h, vcmd)
            # MODIFIED LINE:
            self.sim.setJointTargetForce(h, float(abs(torque)))
        self.sim.step()

    def get_sensor_data(self, show=False):
        rgb, lidar = None, None
        if self.vision_sensor_handle:
            try:
                rx, ry = self.sim.getVisionSensorResolution(self.vision_sensor_handle)
                buf,_,_ = self.sim.getVisionSensorCharImage(self.vision_sensor_handle)
                rgb = np.frombuffer(buf,np.uint8).reshape((ry,rx,3))[::-1]
                if show:
                    cv2.imshow('RGB',rgb)
                    cv2.waitKey(1)
            except:
                pass
        if self.lidar_handle and self.lidar_script:
            _,_,_,buf = self.sim.callScriptFunction(
                'getVelodyneBuffer',self.lidar_script,[],[],[],None)
            if buf:
                cnt = len(buf)//4
                vals = struct.unpack('f'*cnt, buf)
                lidar = np.array(vals,np.float32).reshape(-1,3)
                if show:
                    print(f'[LiDAR] {lidar.shape[0]} pts')
        return rgb, lidar

    def clear_scenery(self): # Renamed for clarity, was clear_towers
        for h in self.scenery_handles:
            try: self.sim.removeObject(h)
            except: pass
        self.scenery_handles = []

    # MODIFIED: spawn_towers is now spawn_scenery and generates mountains
    def spawn_scenery(self,
                     n_items=3,
                     shape_options=8,
                     mountain_target_total_height=1.0,
                     mountain_max_cylinder_height=0.4,
                     mountain_min_cylinder_height=0.05,
                     mountain_peak_radius=0.1,
                     # MODIFIED: Expect a range for base_radius_factor
                     mountain_base_radius_factor_range=(4.0, 7.0), # Example default range
                     mountain_area_bounds_x=None,
                     mountain_area_bounds_y=None,
                     **kwargs
                    ):
        """Spawns mountains using generate_random_mountain with customizable and randomized properties."""
        self.clear_scenery()

        for _ in range(n_items):
            # MODIFIED: Generate a random base_radius_factor for each mountain
            current_base_radius_factor = random.uniform(
                mountain_base_radius_factor_range[0],
                mountain_base_radius_factor_range[1]
            )

            # Prepare arguments for generate_random_mountain
            gen_args = {
                'sim': self.sim,
                'target_total_height': mountain_target_total_height,
                'max_cylinder_height': mountain_max_cylinder_height,
                'min_cylinder_height': mountain_min_cylinder_height,
                'peak_radius': mountain_peak_radius,
                'base_radius_factor': current_base_radius_factor, # Use the randomized factor
                'shape_options': shape_options
            }
            if mountain_area_bounds_x is not None:
                gen_args['area_bounds_x'] = mountain_area_bounds_x
            if mountain_area_bounds_y is not None:
                gen_args['area_bounds_y'] = mountain_area_bounds_y
            
            mountain_handles = generate_random_mountain(**gen_args)
            if mountain_handles:
                self.scenery_handles.extend(mountain_handles)

    def reset_environment(self, sequential=False, **kwargs):
        """Spawn scenery (now mountains) and reset robot to initial pose."""
        if self.robot_base:
            self.sim.setObjectPosition(self.robot_base, -1, self.initial_base_pos)
            self.sim.setObjectOrientation(self.robot_base, -1, self.initial_base_ori)
        
        if sequential:
            # sequential mode (unchanged logic if any)
            pass
        else:
            # MODIFIED: Call spawn_scenery instead of spawn_towers
            self.spawn_scenery(**kwargs)

    def close(self):
        try:
            self.clear_scenery()
            self.sim.stopSimulation()
        except: pass
        cv2.destroyAllWindows()
        print('Simulation closed.')

if __name__ == '__main__':
    class DummySpring:
        def fn_spring(self,q0,q2): return 0.5*q0,0.5*q2

    sim_iface = CoppeliaSimZMQInterface(spring=DummySpring(), exclusion_radius=0.6)
    print("Press 'r' to reset environment (mountains), 'q' to quit.")

    # Example: Initial spawn with customized mountains
    sim_iface.reset_environment(
        n_items=5,  # Number of mountains
        shape_options=9, # CoppeliaSim shape options
        mountain_target_total_height=1.2,
        mountain_max_cylinder_height=0.2,  # Shorter stairs
        mountain_min_cylinder_height=0.05,
        mountain_peak_radius=0.25,         # Wider top
        mountain_base_radius_factor=4.0, # Less tapered
        mountain_area_bounds_x=(-3.0, 3.0), # Custom placement area
        mountain_area_bounds_y=(1.0, 4.0)
    )

    while True:
        u = np.random.uniform(-0.6,0.6,5)
        sim_iface.control(u)
        rgb, lidar = sim_iface.get_sensor_data(show=True)
        if rgb is not None: pass # print('RGB',rgb.shape)
        if lidar is not None: pass # print('LIDAR',lidar.shape[0])
        
        key = cv2.waitKey(1)&0xFF
        if key==ord('r'):
            print('Resetting environment with mountains...')
            sim_iface.reset_environment(
                n_items=5,
                shape_options=9,
                mountain_target_total_height=1.2,
                mountain_max_cylinder_height=0.2,
                mountain_min_cylinder_height=0.05,
                mountain_peak_radius=0.25,
                mountain_base_radius_factor=4.0,
                mountain_area_bounds_x=(-3.0, 3.0),
                mountain_area_bounds_y=(1.0, 4.0)
            )
            print('Environment reset.')
        elif key==ord('q'):
            break
        time.sleep(sim_iface.dt)
    
    sim_iface.close()