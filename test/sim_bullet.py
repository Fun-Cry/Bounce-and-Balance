import numpy as np
import sys
import os

# Add the parent folder to Python path so we can import from env/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.simulationbridge import Sim
from env.robot_model import model_dict
import env.actuator_param as actuator_param

from PIL import Image  # for displaying or saving camera images

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

    # For speed during RL training, you might set direct=True; here we want to see the camera image,
    # so we are using GUI mode (direct=False)
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

    # Optionally, create an output folder for images
    img_folder = os.path.join(os.getcwd(), "imgs")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    for step in range(1000000):
        # u = np.zeros(model["n_a"])  # Zero torque input
        # u = np.random.uniform(-0.1, 0.1, size=model["n_a"])
        u = np.array([-0.5, -0.5, 0, 0, 0])

        X, qa, dqa, contact, tau, *_ = sim.sim_run(u)

        # Instead of getting LiDAR data, let's use the camera image.
        # If your Sim class already has get_camera_image implemented,
        # you can call it to retrieve an RGB image.
        # If it is still commented out in the sim code, you will need to uncomment or implement it.
        if step % 1000 == 0:
            # Retrieve the image from a camera mounted on a link (e.g., link_index 0)
            img = sim.get_camera_image(link_index=0, width=128, height=128, fov=75, near=0.01, far=5.0)
            # Convert the returned image data (assumed to be an array) to a PIL Image object
            # im = Image.fromarray(img)
            # Option 1: Show the image (this will pop up a window, depending on your environment)
            # im.show()
            # Option 2: Save the image to disk
            # img_path = os.path.join(img_folder, f"view_{step:06d}.png")
            # im.save(img_path)
            # print(f"Saved image to {img_path}")

    print("Simulation done.")
