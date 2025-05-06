# test/debug_sim.py
from env.simulation_copp import CoppeliaSimZMQInterface
import numpy as np
import time
import keyboard
import cv2
# import random # Not strictly needed here if randomization is in simulation_copp.py

if __name__ == "__main__":
    class DummySpring:
        def fn_spring(self, q0, q2):
            return 0.5*q0, 0.5*q2

    sim_iface = CoppeliaSimZMQInterface(spring=DummySpring())

    scene_params = {
        'n_items': 1,
        'shape_options': 9,
        'mountain_target_total_height': 1.0,
        'mountain_max_cylinder_height': 0.11,
        'mountain_min_cylinder_height': 0.05,
        'mountain_peak_radius': 0.3,
        # MODIFIED: Provide a range for the base radius factor
        'mountain_base_radius_factor_range': (3.0, 5.0), # Each mountain will get a factor between 2.5 and 5.0
        'mountain_area_bounds_x': (-2.5, 2.5),
        'mountain_area_bounds_y': (1.5, 3.5),
        # Old specific factor (now ignored if range is provided and used by spawn_scenery)
        # 'mountain_base_radius_factor': 3.0, # This line can be removed or will be caught by **kwargs
    }

    print("Initializing environment with custom mountains (random base factor)...")
    # Ensure the key in scene_params matches the parameter name in spawn_scenery
    # If spawn_scenery now takes 'mountain_base_radius_factor_range', use that key.
    # If you kept 'mountain_base_radius_factor' in spawn_scenery and randomized inside,
    # then how you pass it would differ. The code above assumes spawn_scenery takes the range.
    sim_iface.reset_environment(**scene_params)

    print("Simulation started. Press 'r' to reset environment, 'q' to quit.")
    try:
        while True:
            u = np.random.uniform(-1, 1, size=5)
            sim_iface.control(u)
            rgb, lidar = sim_iface.get_sensor_data(show=True)

            if keyboard.is_pressed('r'):
                print("Resetting environment with custom mountains (random base factor)...")
                sim_iface.reset_environment(**scene_params)
                print("Environment reset.")
                time.sleep(0.2) 
            elif keyboard.is_pressed('q'):
                print("Quitting.")
                break

            time.sleep(sim_iface.dt)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sim_iface.close()
        cv2.destroyAllWindows()