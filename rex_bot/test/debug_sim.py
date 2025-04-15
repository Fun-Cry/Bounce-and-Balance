import numpy as np
import sys
import os

# Add the parent folder to Python path so we can import from env/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.simulationbridge import Sim
from env.robot_model import model_dict
import env.actuator_param as actuator_param

class DummySpring:
    def fn_spring(self, q0, q2):
        # Return small spring torque for testing (no spring force)
        return [0.0, 0.0]

if __name__ == "__main__":
    model = model_dict["design_rw"]
    spring = DummySpring()
    q_cal = np.array(model["init_q"])[[0, 2]]

    # Initial state: 13D [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    X_0 = np.zeros(13)
    X_0[2] = model["hconst"]  # Set base height

    sim = Sim(
        X_0=X_0,
        model=model,
        spring=spring,
        q_cal=q_cal,
        dt=1e-3,
        mu=model["mu"],
        direct=False
    )

    print("Simulation starting...")

    for step in range(1000000):
        u = np.zeros(model["n_a"])  # Zero torque input
        X, qa, dqa, contact, tau, *_ = sim.sim_run(u)
        
        # print(1)
        # from PIL import Image
        # Image.fromarray(img).save("view.png")

        if step % 1000 == 0:
            # img = sim.get_camera_image(link_index=0, width=32 , height=64)
            distances = sim.simulate_3d_lidar(link_index=0,
                                            num_azimuth=60,
                                            num_elevation=10,
                                            ray_length=6.0,
                                            visualize=True)
            # print(f"LiDAR 3D shape: {distances.shape}, min: {distances.min():.2f}")




    # sim.close()
    print("Simulation done.")
