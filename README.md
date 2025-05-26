# Reinforcement Learning for Jumping

__Maryia Zhyrko s4093771 and Po-Kai Chen s4283341__

This project focuses on training a one-legged "Rex" hopper robot to jump
 using Reinforcement Learning (RL) with Stable Baselines3 and the
CoppeliaSim robotics simulator. The aim is to develop an agent capable
of stable and effective jumping maneuvers.

## Project Overview

The core of this project involves a custom Gymnasium environment that
interfaces with a CoppeliaSim simulation of the Rex hopper.
Reinforcement learning agents, primarily using Proximal Policy
Optimization (PPO) from Stable Baselines3, are trained to control the
robot's actuators. The project includes functionalities for different
training modes (e.g., `joints_only`), hyperparameter optimization, and
detailed asset management for the robot model.

## Original Model Acknowledgement

The robot model, assets, and initial concept adapted for this project are based on the RExHopper developed by the Robotics Exploration Lab.
* **RExHopper Repository**: [https://github.com/RoboticExplorationLab/RExHopper](https://github.com/RoboticExplorationLab/RExHopper)

## Demo Results

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="demo/raw_input.gif" width="300" alt="Raw Input"/>
        <br>
        <b>Raw Input</b>
      </td>
      <td align="center">
        <img src="demo/actuator.gif" width="300" alt="Actuator Control"/>
        <br>
        <b>Actuator</b>
      </td>
      <td align="center">
        <img src="demo/newest.gif" width="300" alt="Latest Results"/>
        <br>
        <b>Refined Reward</b>
      </td>
    </tr>
  </table>
</div>

## Folder Structure

* **`/` (Root Directory)**
  * `README.md`: This file.
  * `env_runner.py`: Main script for configuring environments, training
    RL agents, and loading models.
  * `train.py`: Potentially an alternative script for running specific
    environment sessions or tests.
  * `hpo.py`: Script for Hyperparameter Optimization of the RL models.
  * `requirements.txt`: Lists the Python dependencies for the project.
  * Various `.zip` files (e.g., `sb3_rex_model*.zip`,
    `jumps_sometimes.zip`, `actuator.zip`): These are various saved model
    checkpoints from different stages of training and experimentation.
* **`/assets`**: Contains 3D model assets for the robot.
  * `hopper_rev08/meshes/*.STL`: Mesh files for the robot parts.
  * `hopper_rev08/urdf/*.urdf, *.csv`: URDF (Unified Robot Description
    Format) files and associated CSVs for the robot model.
* **`/env`**: Core Python package defining the reinforcement learning
  environment.
  * `mountain_env.py`: The custom Gymnasium environment
    (`CoppeliaMountainEnv`) defining observation/action spaces, step logic,
    and the crucial reward function.
  * `simulation_copp.py`: Handles the ZMQ communication interface with
    CoppeliaSim, robot control commands, physics stepping, and sensor data
    retrieval.
  * `actuator.py` & `actuator_param.py`: Define the actuator models
    and their specific parameters.
  * `robot_model.py`: Contains parameters and configurations for the
    robot model itself.
  * `scene_elements.py`: Utility for procedurally generating dynamic
    elements (like 'mountains') within the CoppeliaSim scene.
  * `utils.py`: Utility functions specifically for the environment and
    simulation.
* **`/scenes`**: Contains CoppeliaSim scene files (`.ttt`).
  * `rex_camera.ttt`: Scene configured with a camera sensor.
  * `rex_lidar.ttt`: Scene configured with a LiDAR sensor.
* **`/test`**: Contains scripts for testing and debugging.
  * `debug_sim.py`: For testing and debugging the CoppeliaSim connection
    and basic simulation interaction.
  * `joint_test.py`: For testing individual or groups of robot joints.
* **`/utils`**: Root-level utility scripts.
  * `coppelia_launcher.py`: Script to help automate the launching and
    closing of CoppeliaSim instances.

## Prerequisites

* **Python** (e.g., 3.9 as indicated by `.pyc` files, but likely 3.8+ is
  fine).
* **CoppeliaSim Education Version** (e.g., V4.5.x or V4.6.x). The ZMQ
  Remote API server must be enabled (e.g., `simRemoteApi.start(19997)` in a
  scene script).
* **Python Libraries**: Listed in `requirements.txt`. Key dependencies
  include:
  * `gymnasium`
  * `stable-baselines3[extra]`
  * `numpy`
  * `opencv-python`
  * `coppeliasim-zmqremoteapi-client`

## Setup

1. **Clone the repository.**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt