# !/usr/bin/env/python3

import rclpy
from time import time
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from math import *
import numpy as np

# Q-learning speed parameters
CONST_LATERAL_SPEED_FORWARD = 0.5
CONST_LONGITUDINAL_SPEED_FORWARD = 0.0
CONST_ANGULAR_SPEED_FORWARD = 0.0

CONST_LATERAL_SPEED_LEFT = 0.0
CONST_LONGITUDINAL_SPEED_LEFT = 0.5
CONST_ANGULAR_SPEED_LEFT = 0.0

CONST_LATERAL_SPEED_TURN= 0.0
CONST_LONGITUDINAL_SPEED_TURN = 0.0
CONST_ANGULAR_SPEED_TURN= 0.05

# Feedback control parameters
k_rho = 2
k_alpha = 15
k_beta = -3
v_const = 0.1

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.25
GOAL_ANGLE_THRESHOLD = 5

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q[0], q[1], q[2], q[3]

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# Get theta in [radians]
def getRotation(odomMsg):
    orientation_list = [odomMsg.q_x, odomMsg.q_y, odomMsg.q_z, odomMsg.q_w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return yaw

# Get (x,y) coordinates in [m]
def getPosition(odomMsg):
    x = odomMsg.x
    y = odomMsg.y
    return ( x , y)

# Get linear speed in [m/s]
def getLinVel(odomMsg : Twist):
    return [odomMsg.linear.x, odomMsg.linear.y]


# Get angular speed in [rad/s] - z axis
def getAngVel(odomMsg : Twist):
    return odomMsg.angular.z


# Create ros2 msg Twist()
def createVelMsg(u,v,w):
    velMsg = Twist()
    velMsg.linear.x = u
    velMsg.linear.y = v
    velMsg.angular.z = w
    return velMsg


# Go forward command
def robotGoForward(velPub):
    velMsg = createVelMsg(CONST_LATERAL_SPEED_FORWARD, CONST_LONGITUDINAL_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish_velocity(velMsg)

# Go backward command
def robotGoBackward(velPub):
    velMsg = createVelMsg(-CONST_LATERAL_SPEED_FORWARD, CONST_LONGITUDINAL_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish_velocity(velMsg)

# Turn left command
def robotTurnLeft(velPub):
    velMsg = createVelMsg(CONST_LATERAL_SPEED_LEFT, CONST_LONGITUDINAL_SPEED_LEFT, CONST_ANGULAR_SPEED_LEFT)
    velPub.publish_velocity(velMsg)

# Turn right command
def robotTurnRight(velPub):
    velMsg = createVelMsg(CONST_LATERAL_SPEED_LEFT, -CONST_LONGITUDINAL_SPEED_LEFT, CONST_ANGULAR_SPEED_LEFT)
    velPub.publish_velocity(velMsg)

# Rotate clockwise command
def robotRotateClockwise(velPub):
    velMsg = createVelMsg(CONST_LATERAL_SPEED_TURN, CONST_LONGITUDINAL_SPEED_TURN, -CONST_ANGULAR_SPEED_TURN)
    velPub.publish_velocity(velMsg)

# Rotate counter clockwise command
def robotRotateCounterClockwise(velPub):
    velMsg = createVelMsg(CONST_LATERAL_SPEED_TURN, CONST_LONGITUDINAL_SPEED_TURN, CONST_ANGULAR_SPEED_TURN)
    velPub.publish_velocity(velMsg)

# Stop command
def robotStop(velPub):
    velMsg = createVelMsg(0.0, 0.0, 0.0)
    velPub.publish_velocity(velMsg)

# Set robot position and orientation
def robotSetPose(setPosPub, x, y, theta):
    checkpoint = ModelState()

    checkpoint.model_name = 'GD8'

    checkpoint.pose.position.x = x
    checkpoint.pose.position.y = y
    checkpoint.pose.position.z = 0.0

    [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, radians(theta))

    checkpoint.pose.orientation.x = x_q
    checkpoint.pose.orientation.y = y_q
    checkpoint.pose.orientation.z = z_q
    checkpoint.pose.orientation.w = w_q

    checkpoint.twist.linear.x = 0.0
    checkpoint.twist.linear.y = 0.0
    checkpoint.twist.linear.z = 0.0

    checkpoint.twist.angular.x = 0.0
    checkpoint.twist.angular.y = 0.0
    checkpoint.twist.angular.z = 0.0

    setPosPub.publish_model_state(checkpoint)

    return (x, y, theta)

# Set random initial robot position and orientation
def robotSetRandomPose(setPosPub):
    x_range = np.array([-0.4, 0.6, -0.6, 0.4, 0.2, -0.2, 0.8, -0.8, 0])
    y_range = np.array([-0.4, 0.6, -0.6, 0.4, 0.2, -0.2, 0.8, -0.8, 0])
    theta_range = np.arange(0, 360, 15)

    ind = np.random.randint(0, len(x_range))
    ind_theta = np.random.randint(0, len(theta_range))

    x = x_range[ind]
    y = y_range[ind]
    theta = theta_range[ind_theta]

    checkpoint = ModelState()

    checkpoint.model_name = 'GD8'

    checkpoint.pose.position.x = x
    checkpoint.pose.position.y = y
    checkpoint.pose.position.z = 0.0

    [x_q,y_q,z_q,w_q] = quaternion_from_euler(0.0,0.0,radians(theta))

    checkpoint.pose.orientation.x = x_q
    checkpoint.pose.orientation.y = y_q
    checkpoint.pose.orientation.z = z_q
    checkpoint.pose.orientation.w = w_q

    checkpoint.twist.linear.x = 0.0
    checkpoint.twist.linear.y = 0.0
    checkpoint.twist.linear.z = 0.0

    checkpoint.twist.angular.x = 0.0
    checkpoint.twist.angular.y = 0.0
    checkpoint.twist.angular.z = 0.0

    setPosPub.publish_model_state(checkpoint)
    return ( x , y , theta )

# Perform an action
def robotDoAction(velPub, action):
    status = 'robotDoAction => OK'
    if action == 0:
        robotGoForward(velPub)
    elif action == 1:
        robotGoBackward(velPub)
    elif action == 2:
        robotTurnLeft(velPub)
    elif action == 3:
        robotTurnRight(velPub)
    elif action == 4:
        robotRotateClockwise(velPub)
    elif action == 5:
        robotRotateCounterClockwise(velPub)
    else:
        status = 'robotDoAction => INVALID ACTION'
        robotStop(velPub)

    return status

# Feedback Control Algorithm
def robotFeedbackControl(velPub, x, y, theta, x_goal, y_goal, theta_goal):
    # theta goal normalization
    if theta_goal >= pi:
        theta_goal_norm = theta_goal - 2 * pi
    else:
        theta_goal_norm = theta_goal

    rho = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
    lamda = atan2( y_goal - y , x_goal - x )

    alpha = (lamda -  theta + pi) % (2 * pi) - pi
    beta = (theta_goal - lamda + pi) % (2 * pi) - pi

    if rho < GOAL_DIST_THRESHOLD and degrees(abs(theta-theta_goal_norm)) < GOAL_ANGLE_THRESHOLD:
        status = 'Goal position reached!'
        u = 0
        v = 0
        w = 0
        u_scal = 0
        v_scal = 0
        w_scal = 0
    else:
        status = 'Goal position not reached!'
        u = k_rho * rho * cos(beta)
        v = -k_rho * rho * sin(beta)
        w = k_alpha * alpha + k_beta * beta
        u_scal = u / abs(v) * v_const
        v_scal = v / abs(v) * v_const
        w_scal = w / abs(v) * v_const

    velMsg = createVelMsg(u_scal, v_scal, w_scal)
    velPub.publish(velMsg)

    return status

# Stability Condition
def check_stability(k_rho, k_alpha, k_beta):
    return k_rho > 0 and k_beta < 0 and k_alpha > k_rho

# Strong Stability Condition
def check_strong_stability(k_rho, k_alpha, k_beta):
    return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0