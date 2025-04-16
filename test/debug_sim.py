from env.simulation_copp import CoppeliaSimZMQInterface
import numpy as np

if __name__ == "__main__":
    # Assume you have an instance of a spring model with a fn_spring method.
    # Replace "YourSpringModel" with your actual spring model class or instance.
    class DummySpring:
        def fn_spring(self, q0, q2):
            # Example: simple linear spring compensation.
            k = 0.5
            return [k * q0, k * q2]
    
    spring_instance = DummySpring()
    simInterface = CoppeliaSimZMQInterface(spring=spring_instance)
    
    # Example control loop:
    import time
    try:
        for _ in range(1000):
            # Replace this with your actual control logic.
            u = np.random.uniform(-0.1, 0.1, size=5)
            simInterface.control(u)
            time.sleep(simInterface.dt)
    except KeyboardInterrupt:
        pass
    finally:
        simInterface.close()