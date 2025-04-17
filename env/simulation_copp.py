import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
from env import actuator, actuator_param
import struct

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=1e-3, q_cal=None):
        if q_cal is None:
            q_cal = np.zeros(2)
        self.dt = dt
        self.q_cal = np.array(q_cal)

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        try:
            self.vision_sensor_handle = self.sim.getObject("/base_link_respondable/visionSensor")
            print("✅ Vision sensor found.")
        except Exception:
            self.vision_sensor_handle = None
            print("⚠️ Vision sensor not found.")

        try:
            self.lidar_handle = self.sim.getObject("/base_link_respondable/VelodyneVPL16")
            self.lidar_script = self.sim.getScript(
                self.sim.scripttype_childscript,
                self.lidar_handle
            )
            print("✅ LiDAR and its child script found.")
        except Exception:
            self.lidar_handle = None
            self.lidar_script = None
            print("⚠️ LiDAR or its child script not found.")

        self.sim.setArrayParam(self.sim.arrayparam_gravity, [0, 0, -1])
        if hasattr(self.sim, "setStepping"):
            self.sim.setStepping(True)

        if joint_aliases is None:
            joint_aliases = ["/Joint_0", "/Joint_1", "/Joint_2", "/Joint_3",
                             "/joint_rw0", "/joint_rw1", "/joint_rwz"]
        self.joint_handles = {}
        for alias in joint_aliases:
            try:
                handle = self.sim.getObject(alias)
                self.joint_handles[alias] = handle
                print(f"Retrieved handle for {alias}")
            except Exception as e:
                print(f"Warning: could not get handle for {alias}: {e}")

        self.sim.startSimulation()

        self.spring_fn   = spring.fn_spring
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz= actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)

    def _get_joint_states(self):
        positions, velocities = {}, {}
        for alias, handle in self.joint_handles.items():
            try:
                positions[alias] = self.sim.getJointPosition(handle)
            except Exception as e:
                print(f"Error reading position for {alias}: {e}")
            try:
                velocities[alias] = self.sim.getJointVelocity(handle)
            except Exception as e:
                print(f"Error reading velocity for {alias}: {e}")
                velocities[alias] = 0.0
        return positions, velocities

    def control(self, u):
        u = -u
        positions, velocities = self._get_joint_states()
        if "/Joint_0" not in positions or "/Joint_1" not in positions:
            print("Error: Expected primary joints not found.")
            return

        q0 = positions["/Joint_0"] + self.q_cal[0]
        q2 = positions["/Joint_1"] + self.q_cal[1]
        dq0 = velocities.get("/Joint_0", 0.0)
        dq2 = velocities.get("/Joint_1", 0.0)
        tau_s0, tau_s1 = self.spring_fn(q0=q0, q2=q2)

        tau0, _, _    = self.actuator_q0.actuate(i=u[0], q_dot=dq0)
        tau2, _, _    = self.actuator_q2.actuate(i=u[1], q_dot=dq2)
        tau_rw1, _, _ = self.actuator_rw1.actuate(i=u[2], q_dot=velocities.get("/joint_rw0", 0.0))
        tau_rw2, _, _ = self.actuator_rw2.actuate(i=u[3], q_dot=velocities.get("/joint_rw1", 0.0))
        tau_rwz, _, _ = self.actuator_rwz.actuate(i=u[4], q_dot=velocities.get("/joint_rwz", 0.0))

        commands = {
            "/Joint_0": tau0 + tau_s0,
            "/Joint_1": tau2 + tau_s1,
            "/joint_rw0": tau_rw1,
            "/joint_rw1": tau_rw2,
            "/joint_rwz": tau_rwz
        }
        for alias, torque in commands.items():
            if alias in self.joint_handles:
                vel = 1000.0 if torque>0 else (-1000.0 if torque<0 else 0.0)
                self.sim.setJointTargetVelocity(self.joint_handles[alias], vel)
                self.sim.setJointTargetForce(self.joint_handles[alias], abs(torque))
        self.sim.step()

    def get_sensor_data(self, show=False):
        rgb = depth = lidar = None
        if self.vision_sensor_handle is not None:
            try:
                res_x, res_y = self.sim.getVisionSensorResolution(self.vision_sensor_handle)
                buf, _, _ = self.sim.getVisionSensorCharImage(self.vision_sensor_handle)
                rgb = np.frombuffer(buf, dtype=np.uint8).reshape((res_y, res_x, 3))[::-1,:,:]
                d_buf = self.sim.getVisionSensorDepthBuffer(self.vision_sensor_handle)
                depth = np.flipud(np.array(d_buf, dtype=np.float32).reshape((res_y, res_x)))
                depth = (depth - depth.min())/(depth.max()-depth.min()+1e-8)
                if show:
                    cv2.imshow("RGB", rgb); cv2.imshow("Depth", (depth*255).astype(np.uint8)); cv2.waitKey(1)
            except Exception as e:
                print(f"[Camera] Error: {e}")

        if self.lidar_handle and self.lidar_script:
            # callScriptFunction returns (outInts, outFloats, outStrings, outBuffer)
            outInts, outFloats, outStrings, buffer = self.sim.callScriptFunction(
                'getVelodyneBuffer',
                self.lidar_script,
                [], [], [], None
            )
            if buffer:
                count = len(buffer)//4
                flat = struct.unpack('f'*count, buffer)
                lidar = np.array(flat, dtype=np.float32).reshape(-1, 3)
                if show:
                    print(f"[LiDAR] Retrieved {lidar.shape[0]} points")
            else:
                print("[LiDAR] No data returned.")
                
        return rgb, depth, lidar

    def close(self):
        try:
            self.sim.stopSimulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")
        self.client.disconnect()
        print("Simulation closed.")

if __name__ == "__main__":
    class DummySpring:
        def fn_spring(self, q0, q2):
            return 0.5*q0, 0.5*q2
    simInterface = CoppeliaSimZMQInterface(spring=DummySpring())
    import time
    try:
        for _ in range(1000):
            u = np.array([0.1,0.1,0.05,0.05,0.02])
            simInterface.control(u)
            time.sleep(simInterface.dt)
    except KeyboardInterrupt:
        pass
    finally:
        simInterface.close()