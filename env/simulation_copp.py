import numpy as np
import time
import struct
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from env import actuator, actuator_param

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
        self.exclusion_radius = exclusion_radius

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
            print('⚠️ Could not get base_link_respondable. Towers will not avoid robot.')

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
        self.sim.setArrayParam(self.sim.arrayparam_gravity, [0, 0, -9,81])
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

        # Container for towers
        self.tower_handles = []

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

        mapping = {'/Joint_0':taus[0],'/Joint_1':taus[1],
                   '/joint_rw0':taus[2],'/joint_rw1':taus[3],'/joint_rwz':taus[4]}
        for alias, torque in mapping.items():
            h = self.joint_handles.get(alias)
            if not h: continue
            vcmd = 1000.0 if torque>0 else (-1000.0 if torque<0 else 0.0)
            self.sim.setJointTargetVelocity(h, vcmd)
            self.sim.setJointTargetForce(h, abs(torque))
        self.sim.step()

    def get_sensor_data(self, show=False):
        rgb, lidar = None, None
        # RGB only
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
        # LiDAR
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

    def clear_towers(self):
        for h in self.tower_handles:
            try: self.sim.removeObject(h)
            except: pass
        self.tower_handles = []

    def spawn_towers(self,
                     n_towers=5,
                     area_radius=2.0,
                     radius_range=(0.2,0.5),
                     height_range=(0.1,0.4),
                     shape_options=0):
        """Spawn towers avoiding robot base zone and enable collision."""
        self.clear_towers()
        bx, by = self.initial_base_pos[0], self.initial_base_pos[1]
        for _ in range(n_towers):
            # sample until outside exclusion radius
            for _ in range(20):
                r = np.sqrt(np.random.rand()) * area_radius
                theta = np.random.rand()*2*np.pi
                x, y = bx + r*np.cos(theta), by + r*np.sin(theta)
                if np.hypot(x-bx, y-by) >= self.exclusion_radius:
                    break
            br = float(np.random.uniform(*radius_range))
            h = float(np.random.uniform(*height_range))
            cyl = self.sim.createPrimitiveShape(
                self.sim.primitiveshape_cylinder,[2*br,2*br,h],shape_options)
            self.sim.setObjectPosition(cyl, -1, [x, y, h/2])
            self.sim.setObjectOrientation(cyl, -1, [0,0,0])
            # enable static and respondable
            self.sim.setObjectInt32Param(cyl, self.sim.shapeintparam_static, 1)
            self.sim.setObjectInt32Param(cyl, self.sim.shapeintparam_respondable, 1)
            self.tower_handles.append(cyl)

    def reset_environment(self, sequential=False, **kwargs):
        """Spawn towers and reset robot to initial pose."""
        # reset robot
        if self.robot_base:
            self.sim.setObjectPosition(self.robot_base, -1, self.initial_base_pos)
            self.sim.setObjectOrientation(self.robot_base, -1, self.initial_base_ori)
        # spawn towers
        if sequential:
            # sequential mode (unchanged)
            pass  # existing sequential implementation
        else:
            self.spawn_towers(**kwargs)

    def close(self):
        try: 
            self.clear_towers()
            self.sim.stopSimulation()
        except: pass
        cv2.destroyAllWindows()
        # self.client.disconnect()
        print('Simulation closed.')

if __name__ == '__main__':
    class DummySpring:
        def fn_spring(self,q0,q2): return 0.5*q0,0.5*q2
    sim_iface = CoppeliaSimZMQInterface(spring=DummySpring(), exclusion_radius=0.6)
    print("Press 'r' to reset environment, 'q' to quit.")
    # initial spawn
    sim_iface.reset_environment(
        n_towers=6, area_radius=2.0,
        radius_range=(0.2,0.5), height_range=(0.1,0.4),
        shape_options=9
    )
    while True:
        u = np.random.uniform(-0.6,0.6,5)
        sim_iface.control(u)
        rgb, lidar = sim_iface.get_sensor_data(show=True)
        if rgb is not None: print('RGB',rgb.shape)
        if lidar is not None: print('LIDAR',lidar.shape[0])
        key = cv2.waitKey(1)&0xFF
        if key==ord('r'):
            sim_iface.reset_environment(
                n_towers=6, area_radius=2.0,
                radius_range=(0.2,0.5), height_range=(0.1,0.4),
                shape_options=9
            )
            print('Environment reset.')
        elif key==ord('q'):
            break
        time.sleep(sim_iface.dt)
