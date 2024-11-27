# !/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from lidar import *

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

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def publish_velocity(self, velPub):
        print("Published velocity: ", velPub )
        self.velocity_publisher.publish(velPub)

class LaserSubscriber(Node):
    def __init__(self):
        super().__init__("laser_subscriber")
        self.laser_data = []
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

    def laser_callback(self, msg):
        self.ranges = msg.ranges
        self.laser_data = self.ranges

class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.odom_data = []
        self.odom_sub = self.create_subscription(Odometry, '/odom_rf2o', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        self.odom_data = [self.position.x, self.position.y, self.orientation.z]