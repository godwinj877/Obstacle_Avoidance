# !/usr/bin/env/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from time import time, sleep
from datetime import datetime
import matplotlib.pyplot as plt

import sys

DATA_PATH = '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/QLearn_data'
MODULES_PATH = '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts'
sys.path.insert(0, MODULES_PATH)

from QLearning import *
from lidar import *
from control import *

# Real robot
REAL_ROBOT = False

# Action parameter
MIN_TIME_BETWEEN_ACTION = 0.0

# Initial and goal positions
INIT_POSITIONS_X = [ -0.7, -0.7, -0.5, -1.0, -2.0]
INIT_POSITIONS_Y = [ -0.7, 0.7, 1.0, -2.0, 1.0]
INIT_POSITIONS_THETA = [ 45, -45, -120, -90, 150]
GOAL_POSITIONS_X = [ 2.0, 2.0, 0.5, 1.0, -2.0]
GOAL_POSITIONS_Y = [ 1.0, -1.0, -1.9, 2.0, -1.0,]
GOAL_POSITIONS_THETA = [ 25, -40, -40, 60, -30,]

PATH_IND = 4

# Initial & Goal position
if REAL_ROBOT:
    X_INIT = 0.0
    Y_INIT = 0.0
    THETA_INIT = 0.0
    X_GOAL = 1.5
    Y_GOAL = 1.0
    THETA_GOAL = 30
else:
    RANDOM_INIT_POS = False

    X_INIT = INIT_POSITIONS_X[PATH_IND]
    Y_INIT = INIT_POSITIONS_Y[PATH_IND]
    THETA_INIT = INIT_POSITIONS_THETA[PATH_IND]

    X_GOAL = GOAL_POSITIONS_X[PATH_IND]
    Y_GOAL = GOAL_POSITIONS_Y[PATH_IND]
    THETA_GOAL = GOAL_POSITIONS_THETA[PATH_IND]

# Log file directory - Q table source
Q_TABLE_SOURCE = DATA_PATH + '/Log_learning'

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

 
# Laser Subscriber
class LaserSubscriber(Node):
    def __init__(self):
        super().__init__("laser_subscriber")
        self.ranges = []
        self.angle_increment = 0.0
        self.range_min = 0
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

    def laser_callback(self, msg):
        self.ranges = (msg.ranges)
        self.angle_increment = msg.angle_increment
        self.range_min = msg.range_min

# Velocity Publisher
class VelocityPublisher(Node):
    def __init__(self):
        super().__init__("velocity_publisher")
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def publish_velocity(self, velPub):
        self.velocity_publisher.publish(velPub)

# Model State Publisher
class ModelStatePublisher(Node):
    def __init__(self):
        super().__init__('model_state_publisher')
        self.publisher = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.timer = self.create_timer(1.0, self.publish_model_state)

    def publish_model_state(self, model_state):
        # Set the model state values
        model_state.model_name = 'your_model_name'
        model_state.pose.position.x = 0.0  # Set X position
        model_state.pose.position.y = 0.0  # Set Y position
        model_state.pose.position.z = 0.0  # Set Z position
        model_state.pose.orientation.x = 0.0  # Set X orientation
        model_state.pose.orientation.y = 0.0  # Set Y orientation
        model_state.pose.orientation.z = 0.0  # Set Z orientation
        model_state.pose.orientation.w = 1.0  # Set W orientation

        # Publish the model state
        self.publisher.publish(model_state)
        self.get_logger().info('Publishing model state')
        print("Model State: ", model_state)

if __name__ == '__main__':
    rclpy.init()
    node = rclpy.create_node('control_node')
    rate = node.create_rate(10)

    setPosPub = ModelStatePublisher()
    velPub = VelocityPublisher()

    actions = createActions()
    state_space = createStateSpace()
    Q_table = readQTable(Q_TABLE_SOURCE+'/Qtable.csv')
    print('Initial Q-Table: ')
    print(Q_table)

    # Init time
    t_0 = node.get_clock().now()
    t_start = node.get_clock().now()

    # Init timer
    while not(t_start.to_msg().sec > t_0.to_msg().sec):
        t_start = node.get_clock().now()
    
    t_step = t_start
    count = 0

    # robot in initial position
    robot_in_pos = False

    # because of the video recording
    sleep(5.0)

    while rclpy.ok():
        msgScan = LaserSubscriber()
        odomMsg = PositionSubscriber()

        # Secure the minimum time interval between 2 actions
        step_time = (node.get_clock().now() - t_step).to_msg().sec

        if step_time > MIN_TIME_BETWEEN_ACTION:
            t_step = node.get_clock().now()

            if not robot_in_pos:
                robotStop(velPub)
                # init pos
                if REAL_ROBOT:
                    (x_init, y_init, theta_init) = (0.0,0.0,0.0)
                    rclpy.spin_once(odomMsg)
                    (x, y) = getPosition(odomMsg)
                    theta = degrees(getRotation(odomMsg))
                    robot_in_pos = True
                    print('\r\nInitial position: ')
                    print('x = %.2f [m]' % x)
                    print('y = %.2f [m]' % y)
                    print('theta = %.2f [m]' % theta)
                    print(" ")
                else:
                    if RANDOM_INIT_POS:
                        (x_init, y_init, theta_init) = robotSetRandomPose(setPosPub)
                    else:
                        (x_init, y_init, theta_init) = robotSetPose(setPosPub, X_INIT, Y_INIT, THETA_INIT)

                    # check init pos
                    odomMsg = PositionSubscriber()
                    (x, y) = getPosition(odomMsg)
                    theta = degrees(getRotation(odomMsg))
                    print("x, y, theta init: ", x_init, y_init, theta_init)
                    print("x, y, theta: ", x, y, theta)

                    if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
                        robot_in_pos = True
                        print('\r\nInitial position:')
                        print('x = %.2f [m]' % x)
                        print('y = %.2f [m]' % y)
                        print('theta = %.2f [degrees]' % theta)
                        print('')
                        sleep(1)
                    else:
                        robot_in_pos = False
                    
            else:
                count = count + 1
                text = '\r\nStep %d , Step time %.2f s' % (count, step_time)

                # Get robot position and orientation
                odomMsg = PositionSubscriber()
                ( x , y ) = getPosition(odomMsg)
                theta = getRotation(odomMsg)

                # Get lidar scan
                msgScan = LaserSubscriber()
                ( lidar, angles ) = lidarScan(msgScan)
                ( state_ind, x1, x2 ,x3 ,x4 ,x5 ,x6 ,x7 ,x8) = scanDiscretization(state_space, lidar)

                # Check for objects nearby
                crash = checkCrash(lidar)
                object_nearby = checkObjectNearby(lidar)
                goal_near = checkGoalNear(x, y, X_GOAL, Y_GOAL)
                enable_feedback_control = True

                # Stop the simulation
                if crash:
                    robotStop(velPub)
                    rclpy.shutdown()
                    text = text + ' ==> Crash! End of simulation!'
                    status = 'Crash! End of simulation!'
                # Feedback control algorithm
                elif enable_feedback_control and ( not object_nearby or goal_near ):
                    status = robotFeedbackControl(velPub, x, y, theta, X_GOAL, Y_GOAL, radians(THETA_GOAL))
                    text = text + ' ==> Feedback control algorithm '
                    if goal_near:
                        text = text + '(goal near)'
                # Q-learning algorithm
                else:
                    ( action, status ) = getBestAction(Q_table, state_ind, actions)
                    if not status == 'getBestAction => OK':
                        print('\r\n', status, '\r\n')

                    status = robotDoAction(velPub, action)
                    if not status == 'robotDoAction => OK':
                        print('\r\n', status, '\r\n')
                    text = text + ' ==> Q-learning algorithm'

                text = text + '\r\nx :       %.2f -> %.2f [m]' % (x, X_GOAL)
                text = text + '\r\ny :       %.2f -> %.2f [m]' % (y, Y_GOAL)
                text = text + '\r\ntheta :   %.2f -> %.2f [degrees]' % (degrees(theta), THETA_GOAL)

                if status == 'Goal position reached!':
                    robotStop(velPub)
                    rclpy.shutdown()
                    text = text + '\r\n\r\nGoal position reached! End of simulation!'

                print(text)
                print("\nPublished velocity: ", velPub)
        sleep(3.0)

    node.destroy_node()
    rclpy.shutdown()


