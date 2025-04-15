import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Minimal additional imports to mimic the PyBullet control method.
from env import actuator
from env import actuator_param
# from env import spring

class CoppeliaSimZMQInterface:
    def __init__(self, spring, joint_aliases=None, dt=1e-3, q_cal=None):
        """
        Initialize connection, get joint handles, and start simulation.
        :param joint_aliases: List of aliases/paths for joints (e.g. ['/Joint_0', '/Joint_1', ...]).
        :param dt: Timestep.
        :param q_cal: Calibration values for primary joints (for example, for /Joint_0 and /Joint_1).
        """
        if q_cal is None:
            q_cal = np.zeros(2)
        self.dt = dt
        self.q_cal = np.array(q_cal)
        
        # Create the ZMQ remote client and get the simulation object.
        self.client = RemoteAPIClient()  # Uses default connection settings.
        self.sim = self.client.getObject('sim')
        
        # Enable synchronous stepping if supported.
        if hasattr(self.sim, "setStepping"):
            self.sim.setStepping(True)
        
        # Use default aliases if none are provided.
        if joint_aliases is None:
            joint_aliases = ["/Joint_0", "/Joint_1", "/Joint_2", "/Joint_3",
                             "/joint_rw0", "/joint_rw1", "/joint_rwz"]
        self.joint_aliases = joint_aliases
        self.joint_handles = {}
        
        # Retrieve handles using the new alias notation.
        for alias in self.joint_aliases:
            try:
                # The new recommended method is to use sim.getObject().
                handle = self.sim.getObject(alias)
                self.joint_handles[alias] = handle
                print(f"Retrieved handle for {alias}")
            except Exception as e:
                print(f"Warning: could not get handle for {alias}: {e}")
        
        # Start the simulation.
        self.sim.startSimulation()

        # --- New additions to mimic the PyBullet control dynamics ---
        self.spring_fn = spring.fn_spring
        self.actuator_q0 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2 = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz = actuator.Actuator(dt=self.dt, model=actuator_param.actuator_8318)
        self.spring_fn = spring.fn_spring

    def _get_joint_states(self):
        """
        Retrieve the positions and velocities for all joints.
        Returns:
            positions: dict {alias: position}
            velocities: dict {alias: velocity}
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
        Accept a control vector u (with 5 elements) and send commands to the simulation.
        The mapping follows the PyBullet control logic:
          - /Joint_0 and /Joint_1 are primary joints (with actuator dynamics and spring compensation),
          - /joint_rw0, /joint_rw1, /joint_rwz map to the reaction wheels.
        :param u: numpy array with 5 control inputs.
        """
        # Invert control input to match sign conventions.
        u = -u

        positions, velocities = self._get_joint_states()
        
        # Ensure that primary joints are available.
        if "/Joint_0" not in positions or "/Joint_1" not in positions:
            print("Error: Expected primary joints not found. Check the joint aliases in your scene.")
            return
        
        # Retrieve and calibrate primary joint positions.
        q0 = positions["/Joint_0"] + self.q_cal[0]
        q2 = positions["/Joint_1"] + self.q_cal[1]
        dq_q0 = velocities.get("/Joint_0", 0.0)
        dq_q2 = velocities.get("/Joint_1", 0.0)
        
        # Compute spring torques for the primary joints.
        tau_s = self.spring_fn(q0=q0, q2=q2)
        
        # Compute actuator outputs.
        tau = np.zeros(5)
        # For primary joints
        tau0, _, _ = self.actuator_q0.actuate(i=u[0], q_dot=dq_q0)
        tau2, _, _ = self.actuator_q2.actuate(i=u[1], q_dot=dq_q2)
        tau[0] = tau0 + tau_s[0]
        tau[1] = tau2 + tau_s[1]
        # For reaction wheels
        tau[2], _, _ = self.actuator_rw1.actuate(i=u[2], q_dot=velocities.get("/joint_rw0", 0.0))
        tau[3], _, _ = self.actuator_rw2.actuate(i=u[3], q_dot=velocities.get("/joint_rw1", 0.0))
        tau[4], _, _ = self.actuator_rwz.actuate(i=u[4], q_dot=velocities.get("/joint_rwz", 0.0))
        
        # Map the computed torques to the appropriate joints.
        command_mapping = {
            "/Joint_0": tau[0],
            "/Joint_1": tau[1],
            "/joint_rw0": tau[2],
            "/joint_rw1": tau[3],
            "/joint_rwz": tau[4]
        }
        
        for alias, command in command_mapping.items():
            if alias in self.joint_handles:
                try:
                    self.sim.setJointTargetVelocity(self.joint_handles[alias], command)
                except Exception as e:
                    print(f"Error sending command to {alias}: {e}")
            else:
                print(f"Warning: handle for {alias} not found.")
        
        # Step the simulation since we are in synchronous mode.
        self.sim.step()

    def close(self):
        """
        Stop the simulation and disconnect.
        """
        try:
            self.sim.stopSimulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")
        self.client.disconnect()
        print("Simulation closed.")
