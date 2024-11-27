# !/usr/bin/env/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import matplotlib.pyplot as plt
from datetime import datetime
from sensor_msgs.msg import LaserScan

import sys
sys.path.insert(0, '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts')

from QLearning import *
from lidar import *
from control import *

ANGLE_MAX = int(360 - 1)
ANGLE_MIN = int(1 - 1)
ANGLE_MID = int((ANGLE_MAX + ANGLE_MIN + 1)//2)
HORIZON_WIDTH = int(45)

MIN_TIME_BETWEEN_SCANS = 0
MAX_SIMULATION_TIME = float('inf')


class ScanNode(Node):
    def __init__(self):
        super().__init__('scan_node')
        self.count = 0
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('SCAN NODE START ==> {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        self.t_start = self.get_clock().now()
        self.t = self.t_start

    def scan_callback(self, msg):
        scan_time = (self.get_clock().now() - self.t).to_msg().sec
        sim_time = (self.get_clock().now() - self.t_start).to_msg().sec
        state_space = createStateSpace()
        self.count = self.count + 1

        if scan_time > MIN_TIME_BETWEEN_SCANS:
            lidar, angles = lidarScan(msg)
            state_ind, x1, x2, x3, x4, x5, x6, x7, x8 = scanDiscretization(state_space, lidar)

            crash = checkCrash(lidar)
            object_nearby = checkObjectNearby(lidar)

            self.get_logger().info('Scan cycle: {}'.format(self.count))
            self.get_logger().info('Scan time: {}s'.format(scan_time))
            self.get_logger().info('Simulation time: {}s'.format(sim_time))
            self.get_logger().info('State index: {}'.format(state_ind))
            self.get_logger().info('x1 x2 x3 x4 x5 x6 x7 x8')
            self.get_logger().info('{} {} {} {} {} {} {} {}'.format(x1, x2, x3, x4, x5, x6, x7, x8))
            if crash:
                self.get_logger().info('CRASH!')
            if object_nearby:
                self.get_logger().info('OBJECT NEARBY!')

            lidar = np.array(lidar)
            print(lidar, len(lidar))
            lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1], 
                                            lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1],
                                            lidar[(ANGLE_MID - HORIZON_WIDTH):(ANGLE_MID):1], 
                                            lidar[(ANGLE_MID):(ANGLE_MID + HORIZON_WIDTH):1]))
            angles_horizon = np.linspace(90 + HORIZON_WIDTH, 90 - HORIZON_WIDTH, 150)

            x_horizon = np.array([r * np.cos(np.radians(theta)) for r, theta in zip(lidar_horizon, angles_horizon)])
            y_horizon = np.array([r * np.sin(np.radians(theta)) for r, theta in zip(lidar_horizon, angles_horizon)])

            plt.figure(1)
            plt.clf()
            plt.xlabel('distance[m]')
            plt.ylabel('distance[m]')
            plt.xlim((-1.0, 1.0))
            plt.ylim((-0.2, 1.2))
            plt.title('Lidar horizon')
            plt.axis('equal')
            plt.plot(x_horizon, y_horizon, 'b.', markersize=8, label='obstacles')
            plt.plot(0, 0, 'r*', markersize=20, label='robot')
            plt.legend(loc='lower right', shadow=True)
            plt.draw()
            plt.pause(0.0001)

            self.t = self.get_clock().now()

        if sim_time > MAX_SIMULATION_TIME:
            self.get_logger().info('SCAN NODE STOP ==> {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    scan_node = ScanNode()
    rclpy.spin(scan_node)

if __name__ == '__main__':
    main()
