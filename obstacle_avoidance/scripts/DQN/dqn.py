# !/usr/bin/env/python3

import rclpy
import rclpy.node as Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import memory
import random

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# Position Subscriber
class PositionSubscriber(Node):
    def __init__(self):
        super().__init__("position_subscriber")
        self.x = 0
        self.y = 0
        self.q_x = 0
        self.q_y = 0
        self.q_z = 0
        self.q_w = 0
        self.position_sub = self.create_subscription(Odometry, "/odom", self.pos_callback, 10)

    def pos_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.q_x = msg.pose.pose.orientation.x
        self.q_y = msg.pose.pose.orientation.y
        self.q_z = msg.pose.pose.orientation.z
        self.q_w = msg.pose.pose.orientation.w
        print("Position: ", self.x, self.y)
 
# Laser Subscriber
class LaserSubscriber(Node):
    def __init__(self):
        super().__init__("laser_subscriber")
        self.ranges = []
        self.angle_increment = 0.0
        self.range_min = 0
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

    def laser_callback(self, msg):
        print("Entered laser callback")
        self.ranges = msg.ranges
        self.angle_increment = msg.angle_increment
        self.range_min = msg.range_min

# Velocity Publisher
class VelocityPublisher(Node):
    def __init__(self):
        super().__init__("velocity_publisher")
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def publish_velocity(self, velPub):
        print("Published velocity: ", velPub )
        self.velocity_publisher.publish(velPub)

# Gazebo Resetter
class GazeboWorldResetter(Node):
    def __init__(self):
        super().__init__('gazebo_world_resetter')
        # Create a client for the reset_simulation service
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        # Wait for the service to be available
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /reset_simulation not available, waiting...')
        self.call_reset_simulation_service()

    def call_reset_simulation_service(self):
        # Create an empty request
        request = Empty.Request()
        # Call the reset_simulation service
        future = self.reset_simulation_client.call_async(request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, future)
        # Check if the service call was successful
        if future.result() is not None:
            self.get_logger().info('Gazebo world reset successful')
        else:
            self.get_logger().error('Failed to reset Gazebo world')

class ReplayBuffer(object):
    def __init__(self, capacity):
        super().__init__()
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(Node):
    def __init__(self, input_size, output_size, memory_size, discountFactor, learningRate, learnStart):
        self.input_size = input_size
        self.output_size = output_size
        self.memory = memory.Memory(memory_size)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.learnStart = learnStart
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initNetworks(self, n_observations, n_actions, training=True, modelPath = None):
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x
    
class MecanumWheel(Node):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000
        self.tau = 0.005
        self.learningRate = 1e-4

        self.actions = [0, 1, 2, 3, 4, 5]
        self.n_actions = len(self.actions)
        self.laser_obs = LaserSubscriber()
        self.pos_obs = PositionSubscriber()
        self.observations = [self.laser_obs, self.pos_obs]
        self.n_observations = len(self.observations)


if __name__ == "__main__":
    rclpy.init()
    node = DQN()
    rclpy.spin(node)
    node.destory_node()
    rclpy.shutdown()

