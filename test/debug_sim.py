from env.simulation_copp import CoppeliaSimZMQInterface
import numpy as np
import time
import keyboard
import cv2

if __name__ == "__main__":
    # Dummy spring model for torque compensation
    class DummySpring:
        def fn_spring(self, q0, q2):
            return 0.5*q0, 0.5*q2

    sim_iface = CoppeliaSimZMQInterface(spring=DummySpring())
    # initial random towers
    sim_iface.reset_environment(
        n_towers=16,
        area_radius=3.0,
        radius_range=(0.1, 0.3),
        height_range=(0.1, 0.25),
        shape_options=9 # back-face cull + respondable
    )

    print("Press 'r' to reset environment, 'q' to quit.")
    try:
        while True:
            # example random control, replace with policy output
            u = np.random.uniform(-1, 1, size=5)
            sim_iface.control(u)
            rgb, lidar = sim_iface.get_sensor_data(show=True)

            # if rgb is not None:
            #     print(f"RGB shape: {rgb.shape}")
            # if lidar is not None:
            #     print(f"LiDAR points: {lidar.shape[0]}")

            # key = cv2.waitKey(1) & 0xFF。
            if keyboard.is_pressed('r'):
                sim_iface.reset_environment(
                    n_towers=16,
                    area_radius=3.0,
                    radius_range=(0.1, 0.3),
                    height_range=(0.1, 0.25),
                    shape_options=9
                )
                print("Environment reset.")
            elif keyboard.is_pressed('q'):
                print("Quitting.")
                break

            time.sleep(sim_iface.dt)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sim_iface.close()
        cv2.destroyAllWindows()