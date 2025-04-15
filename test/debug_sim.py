import numpy as np
import sys
import os

# Add the parent folder to Python path so we can import from env/
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.robot_model import model_dict
import env.actuator_param as actuator_param

# Import the updated CoppeliaSim interface.
from env.simulation_copp import CoppeliaSimZMQInterface

class DummySpring:
    def fn_spring(self, q0, q2):
        # Dummy spring function returning zero torque for testing.
        return [0.0, 0.0]

if __name__ == "__main__":
    # Select your robot design from the model dictionary.
    model = model_dict["design_rw"]
    spring = DummySpring()
    
    # Calibration: get calibration values from your model.
    q_cal = np.array(model["init_q"])[[0, 2]]
    
    # Define joint aliases as assigned in your CoppeliaSim scene.
    joint_aliases = ["/Joint_0", "/Joint_1", "/Joint_2", "/Joint_3",
                     "/joint_rw0", "/joint_rw1", "/joint_rwz"]
    
    # Create an instance of the CoppeliaSim interface.
    sim_interface = CoppeliaSimZMQInterface(
        spring,
        joint_aliases=joint_aliases,
        dt=1e-3,
        q_cal=q_cal
    )
    
    print("Simulation starting...")
    
    # Run a control loop that sends random control vectors.
    num_steps = 1000000
    for step in range(num_steps):
        # Generate a random control vector (5 elements, one per actuator).
        u = 0.1 * np.random.randn(model["n_a"])
        sim_interface.control(u)
        
        if step % 1000 == 0:
            print(f"Step {step}: control vector: {u}")
    
    sim_interface.close()
    print("Simulation done.")
