import pybullet as p
import pybullet_data
import math
import random
import time

def create_staircase():
    """
    Creates a single contiguous staircase made of multiple steps.
    
    Two modes are available:
      • 'diagonal': A long, continuous path in a fixed (diagonal) direction.
      • 'curved': A staircase whose heading changes randomly from step to step.
      
    Each step is positioned such that its bottom touches the top surface of the previous step.
    """
    # Decide randomly whether to generate a diagonal (fixed direction) or a curved staircase.
    mode = random.choice(['diagonal', 'curved'])
    
    # Randomly choose the total number of steps.
    num_steps = random.randint(8, 15)
    
    # Randomly choose common parameters for all steps.
    step_depth = random.uniform(0.8, 1.2)   # Horizontal depth (front-to-back)
    step_width = random.uniform(1.0, 2.0)     # Lateral width
    step_height = random.uniform(0.2, 0.4)    # Vertical height per step
    
    # The "base" is the current bottom surface of the evolving staircase.
    # Start at a random position on the ground.
    current_base = [random.uniform(-2, 2), random.uniform(-2, 2), 0]
    # current_base = [random.uniform(5, 8), random.uniform(5, 8), 0]

    
    # Choose the initial heading angle (in radians).
    if mode == 'diagonal':
        # Fixed 45° angle (diagonal) for a long, straight ramp.
        current_angle = math.pi / 4
        print("Creating a long diagonal staircase (fixed 45° path).")
    else:
        # Otherwise, use a random initial angle.
        current_angle = random.uniform(0, 2 * math.pi)
        print("Creating a curved, non-linear staircase.")
    
    for i in range(num_steps):
        # For the curved mode, randomly adjust the heading a bit for each step.
        if mode == 'curved':
            delta_angle = random.uniform(-math.pi/6, math.pi/6)  # Change by up to ±30°
            current_angle += delta_angle
        
        # To place the current step, compute its center.
        # The step will be placed so that its bottom surface touches the current base.
        # For a box step, the center is offset upward by half the step's height...
        # and forward along the current direction by half the step's depth.
        dx_center = (step_depth / 2) * math.cos(current_angle)
        dy_center = (step_depth / 2) * math.sin(current_angle)
        dz_center = step_height / 2  # half the vertical height
        
        center_pos = [
            current_base[0] + dx_center,
            current_base[1] + dy_center,
            current_base[2] + dz_center
        ]
        
        # Define half-extents for the box (each step).
        half_extents = [step_depth / 2, step_width / 2, step_height / 2]
        
        # Assign a color that gradually changes over the staircase.
        color = [0.2 + (i/num_steps)*0.8, 0.3, 1 - (i/num_steps)*0.7, 1]
        
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=collision_shape,
                          baseVisualShapeIndex=visual_shape,
                          basePosition=center_pos)
        
        print(f"Step {i+1}/{num_steps}: Center = {center_pos}")
        
        # Update the "base" for the next step.
        # Since we want the steps to be contiguous, the next step’s bottom starts
        # exactly one full step ahead in the current direction and with an added vertical lift.
        dx_base = step_depth * math.cos(current_angle)
        dy_base = step_depth * math.sin(current_angle)
        dz_base = step_height
        
        current_base = [
            current_base[0] + dx_base,
            current_base[1] + dy_base,
            current_base[2] + dz_base
        ]

def reset_and_create_environment():
    """
    Resets the PyBullet simulation and creates a ground plane along with the non-straight staircase.
    """
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    create_staircase()

def main():
    # Connect to the PyBullet physics server in GUI mode.
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Initialize the environment.
    reset_and_create_environment()
    print("\nEnvironment loaded. Press the 'r' key in the PyBullet window to generate a new staircase.")
    
    time_step = 1.0 / 240.0
    while p.isConnected():
        p.stepSimulation()
        time.sleep(time_step)
        
        # Check keyboard events.
        keys = p.getKeyboardEvents()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            print("\n'r' key pressed. Resetting simulation and generating a new staircase...\n")
            reset_and_create_environment()

if __name__ == "__main__":
    main()
