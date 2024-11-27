import rclpy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import time
from sensor_msgs.msg import JointState
from DDPG.lidar import *

# from DDPG.topic import *


def joint_states_callback(msg):
    # Extract and print joint positions from the JointState message
    if 'position' in msg.__slots__:
        print("Joint Positions:")
        for joint_name, position in zip(msg.name, msg.position):
            print(f"  {joint_name}: {position}")
    else:
        print("Joint positions not available in the JointState message.")

def odom_callback(msg):
    # Extract and print the pose information from the Odometry message
    pose = msg.pose.pose
    position = pose.position
    orientation = pose.orientation
    print("Odometry Data:")
    print(f"Position: x={position.x:.5f}, y={position.y:.5f}, z={position.z:.5f}")
    #print(f"Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}\n")

def scan_callback(msg):
    # Extract and print laser scan information
    ranges = msg.ranges
    print("Laser Scan Data:")
    lidar_data = scanDiscretization(ranges)
    print(lidar_data)
    print('\n')
    # print(f"  Number of ranges: {len(ranges)}")
    # print(f"  Min Range: {min(ranges)}")
    # print(f"  Max Range: {max(ranges)}")

def main():
    rclpy.init()

    # Create a ROS 2 node
    node = rclpy.create_node('subscriber_node')

    # Create a subscription to the /odom topic with the Odometry message type
    # odom_subscription = node.create_subscription(Odometry, '/odom_rf2o', odom_callback, 10)
    # odom_subscription  # prevent unused variable warning

    # Create a subscription to the /scan topic with the LaserScan message type
    scan_subscription = node.create_subscription(LaserScan, '/scan', scan_callback, 10)
    scan_subscription  # prevent unused variable warning

    # Create a subscription to the /joint_states topic with the JointState message type
    # joint_states_subscription = node.create_subscription(JointState, '/joint_states', joint_states_callback, 10)
    # joint_states_subscription

    # Display a message indicating that the node is subscribed to /odom and /scan
    node.get_logger().info('Subscribed to /odom and /scan topics')

    try:
        # Spin the node to keep it running and receiving messages
        while(1):
            rclpy.spin_once(node)
            time.sleep(0.5)
    except KeyboardInterrupt:
        # Handle keyboard interrupt by shutting down the node
        pass
    finally:
        # Clean up resources before exiting
        node.destroy_node()
        rclpy.shutdown()





if __name__ == '__main__':
    main()
