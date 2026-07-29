# Obstacle Avoidance for Autonomous Mobile Robots using Reinforcement Learning

## Overview

This project presents a reinforcement learning-based obstacle avoidance framework for an autonomous mobile robot operating in a simulated environment. The objective is to enable the robot to navigate safely while avoiding obstacles and reaching predefined goals without relying on manually designed navigation rules.

The project is developed using **ROS2**, **Gazebo**, and **Python**, providing a modular framework for training and evaluating reinforcement learning algorithms in realistic robotic environments.

---

## Objectives

- Develop an autonomous obstacle avoidance system using reinforcement learning.
- Enable collision-free navigation in unknown environments.
- Train and evaluate navigation policies in simulation before deployment.
- Provide a modular framework that can be extended to different reinforcement learning algorithms.

---

## Features

- Autonomous mobile robot simulation
- Reinforcement learning-based navigation
- Real-time LiDAR perception
- Goal-directed motion
- Collision avoidance
- ROS2 and Gazebo integration
- Modular training framework
- Custom simulation environments

---

## System Architecture

The navigation pipeline consists of the following components:

1. Gazebo simulation environment
2. Mobile robot model
3. LiDAR sensor for obstacle perception
4. Reinforcement learning agent
5. Motion controller
6. Navigation feedback loop

The robot continuously observes its environment, selects an action using the learned policy, executes the action, and receives a reward based on its performance.

---

## Repository Structure

```
Obstacle_Avoidance/

├── obstacle_avoidance/
│   ├── launch/
│   ├── config/
│   ├── worlds/
│   ├── rviz/
│   ├── scripts/
│   └── src/
│
├── gazebo_mecanum_plugins/
│
├── LICENSE
└── README.md
```

---

## Software Stack

- Python
- ROS2
- Gazebo
- OpenCV
- NumPy
- Reinforcement Learning

---

## Robot Perception

The robot uses LiDAR measurements to perceive nearby obstacles and estimate free space for navigation.

Sensor information is processed to generate the observation space used by the reinforcement learning agent.

---

## Reinforcement Learning Framework

### Observation Space

The agent receives information describing the current navigation state, including environmental observations obtained from onboard sensors.

Typical observations include:

- LiDAR measurements
- Robot orientation
- Goal information
- Navigation state

### Action Space

The agent predicts continuous motion commands for the robot, including:

- Linear velocity
- Angular velocity

### Reward Design

The reward function encourages the robot to:

- Reach the navigation goal
- Avoid collisions
- Reduce unnecessary movements
- Follow efficient trajectories

---

## Simulation Environment

Training and evaluation are performed in Gazebo using custom environments containing static obstacles.

The framework can be extended to more complex environments including dynamic obstacles and multi-goal navigation tasks.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/godwinj877/Obstacle_Avoidance.git
```

Build the workspace.

```bash
colcon build
```

Source the workspace.

```bash
source install/setup.bash
```

Launch the simulation.

```bash
ros2 launch obstacle_avoidance <launch_file>.launch.py
```

Run the reinforcement learning agent.

```bash
python3 <training_script>.py
```

Replace the placeholders above with the actual launch file and training script names used in the repository.

---

## Results

The reinforcement learning agent successfully learns navigation policies that enable the robot to:

- Navigate toward target locations
- Avoid collisions with obstacles
- Improve navigation efficiency over training
- Generate smooth motion commands

Future versions of this repository will include quantitative evaluation metrics, learning curves, and navigation performance comparisons.

---

## Future Work

- Compare multiple reinforcement learning algorithms
- Dynamic obstacle avoidance
- Multi-goal navigation
- Sim-to-real deployment
- Sensor fusion
- Path planning integration
- Performance benchmarking

---

## References

- Robot Operating System (ROS2)
- Gazebo Simulator
- Reinforcement Learning
- Deep Reinforcement Learning for Robot Navigation

---

## Author

**Godwin Joseph**

AI/ML Engineer

B.Tech. Mechanical Engineering, IIT Palakkad

Areas of Interest:
Artificial Intelligence • Reinforcement Learning • Robotics • Computer Vision • Autonomous Systems

---

## License

This project is licensed under the Apache 2.0 License.
