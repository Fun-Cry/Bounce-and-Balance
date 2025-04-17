from env.simulation_copp import CoppeliaSimZMQInterface
import numpy as np
import time
import cv2

if __name__ == "__main__":
    class DummySpring:
        def fn_spring(self, q0, q2):
            k = 0.5
            return [k * q0, k * q2]

    spring_instance = DummySpring()
    simInterface = CoppeliaSimZMQInterface(spring=spring_instance)

    try:
        for _ in range(1000):
            # Random control input (replace with your policy)
            u = np.random.uniform(-0.1, 0.1, size=5)
            simInterface.control(u)

            # Unified sensor reading
            rgb, depth, lidar = simInterface.get_sensor_data(show=True)

            if depth is not None:
                print(f"Depth shape: {depth.shape}")

            if rgb is not None:
                print(f"RGB shape: {rgb.shape}")

            if lidar is not None:
                print(f"LiDAR points: {lidar.shape[0]}")

            time.sleep(simInterface.dt)

    except KeyboardInterrupt:
        pass
    finally:
        simInterface.close()
        cv2.destroyAllWindows()
