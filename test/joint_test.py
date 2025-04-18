import numpy as np
import time
import cv2
import keyboard

from env.simulation_copp import CoppeliaSimZMQInterface

if __name__ == "__main__":
    # Dummy spring for torque compensation
    class DummySpring:
        def fn_spring(self, q0, q2):
            return 0.5*q0, 0.5*q2

    sim_iface = CoppeliaSimZMQInterface(spring=DummySpring())

    # Spawn initial set of low, close towers
    sim_iface.reset_environment(
        n_towers=16,
        area_radius=3.0,
        radius_range=(0.1, 0.3),
        height_range=(0.1, 0.25),
        shape_options=9
    )

    print("Randomly moving Joint_0 and Joint_1. Press 'q' to quit.")

    try:
        while True:
            # Check for quit key
            if keyboard.is_pressed('q'):
                print("Quitting random joint movement.")
                break

            # Random torque for hip (joint 0) and knee (joint 1), zeros elsewhere
            u = np.zeros(5, dtype=np.float64)
            u[0] = np.random.uniform(-0.1, 0.1)  # hip
            u[1] = np.random.uniform(-0.1, 0.1)  # knee

            # Apply control
            sim_iface.control(u)

            # Read back joint positions
            pos, _ = sim_iface._get_joint_states()
            j0 = pos.get("/Joint_0", float('nan'))
            j1 = pos.get("/Joint_1", float('nan'))
            print(f"Joint_0: {j0:.3f} rad, Joint_1: {j1:.3f} rad")

            # small delay
            time.sleep(sim_iface.dt)

    except KeyboardInterrupt:
        pass
    finally:
        # zero torques, stop sim
        sim_iface.control(np.zeros(5))
        sim_iface.close()
        cv2.destroyAllWindows()
