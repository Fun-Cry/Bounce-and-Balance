import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import cv2
# Minimal additional imports to mimic the PyBullet control method.
from env import actuator
from env import actuator_param
# 'spring' should be provided as an instance with a method fn_spring
import struct

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=1e-3, q_cal=None):
        """
        Initialize connection, retrieve joint handles, and start the simulation.
        
        Note: The dummy objects and their link constraint between Link 1 and Link 3
        must be set manually in the scene (using the GUI).
        
        :param spring: An instance with a method fn_spring for spring compensation.
        :param joint_aliases: List of joint alias strings (e.g., ['/Joint_0', '/Joint_1', ...]).
        :param dt: Time step for simulation.
        :param q_cal: Calibration values for the primary joints.
        """
        if q_cal is None:
            q_cal = np.zeros(2)
        self.dt = dt
        self.q_cal = np.array(q_cal)
        
        # Create the remote client and retrieve the simulation object.
        self.client = RemoteAPIClient()  # Uses default connection settings.
        self.sim = self.client.getObject('sim')
        # self.vision_sensor_handle = self.sim.getObject("/base_link_respondable/visionSensor")
        try:
            self.vision_sensor_handle = self.sim.getObject("/base_link_respondable/visionSensor")
            print("✅ Vision sensor found.")
        except Exception:
            self.vision_sensor_handle = None
            print("⚠️ Vision sensor not found.")

        # Try to get LiDAR
        try:
            self.lidar_handle = self.sim.getObject("/base_link_respondable/VelodyneVPL16")
            self.lidar_script = self.sim.getScript(
                self.sim.scripttype_childscript,
                self.lidar_handle
            )
            # print("✅ LiDAR child‑script handle:", self.lidar_script)
            print("✅ LiDAR found.")
        except Exception:
            self.lidar_handle = None
            print("⚠️ LiDAR not found.")
            
                
    # sim.scripttype_childscript is the right type
    
        
        # For testing, we set gravity to [0, 0, 0]. (Change if needed.)
        self.sim.setArrayParam(self.sim.arrayparam_gravity, [0, 0, -1])
        
        # Enable synchronous stepping if supported.
        if hasattr(self.sim, "setStepping"):
            self.sim.setStepping(True)
        
        # Use default joint aliases if none provided.
        if joint_aliases is None:
            joint_aliases = ["/Joint_0", "/Joint_1", "/Joint_2", "/Joint_3",
                             "/joint_rw0", "/joint_rw1", "/joint_rwz"]
        self.joint_aliases = joint_aliases
        self.joint_handles = {}
        
        # Retrieve joint handles using the provided aliases.
        for alias in self.joint_aliases:
            try:
                handle = self.sim.getObject(alias)
                self.joint_handles[alias] = handle
                print(f"Retrieved handle for {alias}")
            except Exception as e:
                print(f"Warning: could not get handle for {alias}: {e}")
        
        # Start the simulation.
        self.sim.startSimulation()
        
        # Set up actuator dynamics to mimic PyBullet.
        self.spring_fn = spring.fn_spring
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)
        self.spring_fn = spring.fn_spring

        # Note: The dummy-dummy constraint between Link 1 and Link 3 is assumed to be set manually in the scene.
    
    def _get_joint_states(self):
        """
        Retrieve joint positions and velocities.
        
        Returns:
            positions: A dictionary mapping joint alias to its position.
            velocities: A dictionary mapping joint alias to its velocity.
        """
        positions = {}
        velocities = {}
        for alias, handle in self.joint_handles.items():
            try:
                pos = self.sim.getJointPosition(handle)
                positions[alias] = pos
            except Exception as e:
                print(f"Error reading position for {alias}: {e}")
            try:
                vel = self.sim.getJointVelocity(handle)
                velocities[alias] = vel
            except Exception as e:
                print(f"Error reading velocity for {alias}: {e}")
                velocities[alias] = 0.0
        return positions, velocities

    def control(self, u):
        """
        Accept a control vector u (with 5 elements) and send torque commands to the simulation.
        The method mimics torque control by setting a high target velocity and using
        setJointTargetForce to limit the applied force.
        
        :param u: numpy array of 5 control inputs.
        """
        # Invert control input to match sign conventions.
        u = -u

        positions, velocities = self._get_joint_states()
        
        # Check for primary joint states.
        if "/Joint_0" not in positions or "/Joint_1" not in positions:
            print("Error: Expected primary joints not found. Check joint aliases.")
            return
        
        # Calibrate and retrieve primary joint positions.
        q0 = positions["/Joint_0"] + self.q_cal[0]
        q2 = positions["/Joint_1"] + self.q_cal[1]
        dq_q0 = velocities.get("/Joint_0", 0.0)
        dq_q2 = velocities.get("/Joint_1", 0.0)
        
        # Compute spring torques.
        tau_s = self.spring_fn(q0=q0, q2=q2)
        
        # Compute actuator outputs.
        tau = np.zeros(5)
        tau0, _, _ = self.actuator_q0.actuate(i=u[0], q_dot=dq_q0)
        tau2, _, _ = self.actuator_q2.actuate(i=u[1], q_dot=dq_q2)
        tau[0] = tau0 + tau_s[0]
        tau[1] = tau2 + tau_s[1]
        tau[2], _, _ = self.actuator_rw1.actuate(i=u[2], q_dot=velocities.get("/joint_rw0", 0.0))
        tau[3], _, _ = self.actuator_rw2.actuate(i=u[3], q_dot=velocities.get("/joint_rw1", 0.0))
        tau[4], _, _ = self.actuator_rwz.actuate(i=u[4], q_dot=velocities.get("/joint_rwz", 0.0))
        
        # Map computed torques to the corresponding joints.
        command_mapping = {
            "/Joint_0": tau[0],
            "/Joint_1": tau[1],
            "/joint_rw0": tau[2],
            "/joint_rw1": tau[3],
            "/joint_rwz": tau[4]
        }
        
        # Apply torque control by setting a high target velocity (to "drive" the joint)
        # and limiting the force via setJointTargetForce.
        for alias, torque_command in command_mapping.items():
            if alias in self.joint_handles:
                try:
                    target_velocity = 0.0
                    if abs(torque_command) > 1e-6:
                        target_velocity = 1e3 if torque_command > 0 else -1e3
                    self.sim.setJointTargetVelocity(self.joint_handles[alias], target_velocity)
                    self.sim.setJointTargetForce(self.joint_handles[alias], abs(torque_command))
                    print(f"Set joint {alias}: target velocity = {target_velocity}, force = {abs(torque_command)}")
                except Exception as e:
                    print(f"Error sending command to {alias}: {e}")
            else:
                print(f"Warning: handle for {alias} not found.")
        
        # Step the simulation (since we're in synchronous mode).
        self.sim.step()

    def close(self):
        """
        Stop the simulation and disconnect the remote client.
        """
        try:
            self.sim.stopSimulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")
        self.client.disconnect()
        print("Simulation closed.")
        
    def get_sensor_data(self, show=False):
        """
        Returns (rgb, depth, lidar):
        - rgb: H×W×3 uint8 image or None
        - depth: H×W float32 normalized [0,1] or None
        - lidar: N×3 float32 point cloud or None
        """
        rgb, depth, lidar = None, None, None

        # --- camera (unchanged) ---
        if self.vision_sensor_handle is not None:
            try:
                res_x, res_y = self.sim.getVisionSensorResolution(self.vision_sensor_handle)
                img_buf, _, _ = self.sim.getVisionSensorCharImage(self.vision_sensor_handle)
                rgb = np.frombuffer(img_buf, dtype=np.uint8).reshape((res_y, res_x, 3))
                rgb = np.flipud(rgb)

                depth_buf = self.sim.getVisionSensorDepthBuffer(self.vision_sensor_handle)
                depth = np.array(depth_buf, dtype=np.float32).reshape((res_y, res_x))
                depth = np.flipud(depth)
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

                if show:
                    cv2.imshow("RGB", rgb)
                    cv2.imshow("Depth", (depth * 255).astype(np.uint8))
                    cv2.waitKey(1)
            except Exception as e:
                print(f"[Camera] Error reading: {e}")

        # --- LiDAR (new) ---
        # In your get_sensor_data method, modify the LiDAR section:
        if self.lidar_handle is not None:
            try:
                # The correct format is "functionName@objectPath"
                res = self.sim.callScriptFunction(
                    'getVelodyneBuffer@/base_link_respondable/VelodyneVPL16',
                    self.sim.scripttype_childscript,
                    # -1,  # Use -1 to indicate the function is called by name
                    # [], [], [], b''
                )
                
                packed = res[3]  # The packed buffer is in the fourth return value
                if packed:
                    count = len(packed) // 4
                    flat = struct.unpack('<' + 'f'*count, packed)
                    pts4 = np.array(flat, dtype=np.float32).reshape(-1, 4)
                    lidar = pts4[:, :3]
                    if show:
                        print(f"[LiDAR] Retrieved {lidar.shape[0]} points")
            except Exception as e:
                print(f"[LiDAR] Error reading: {e}")

        return rgb, depth, lidar



# Example usage:
if __name__ == "__main__":
    # Create a dummy spring model with a simple linear spring compensation.
    class DummySpring:
        def fn_spring(self, q0, q2):
            k = 0.5
            return [k * q0, k * q2]
    
    spring_instance = DummySpring()
    simInterface = CoppeliaSimZMQInterface(spring=spring_instance)
    
    import time
    try:
        for _ in range(1000):
            u = np.array([0.1, 0.1, 0.05, 0.05, 0.02])
            simInterface.control(u)
            time.sleep(simInterface.dt)
    except KeyboardInterrupt:
        pass
    finally:
        simInterface.close()
