# !/usr/bin/env/python3

import rclpy
from rclpy.qos import QoSProfile
from time import time,sleep
from datetime import datetime
import matplotlib.pyplot as plt

import sys
DATA_PATH = '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/QLearn_data'
MODULES_PATH = '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts'
sys.path.insert(0, MODULES_PATH)

from control import *

X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0
X_GOAL = 3.0
Y_GOAL = 2.0
THETA_GOAL = 15.0

# init trajectory
X_traj = np.array([])
Y_traj = np.array([])
THETA_traj = np.array([])
X_goal = np.array([])
Y_goal = np.array([])
THETA_goal = np.array([])

# log directory
LOG_DIR = DATA_PATH + '/Log_feedback'

def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.q = msg.pose.pose.orientation

if __name__ == '__main__':
    rclpy.init(None)
    node = rclpy.create_node('feedback_control_node')

    # init topics
    setPosPub = node.create_publisher(ModelState, '/gazebo/set_model_state', QoSProfile(depth=10))
    velPub = node.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10))

    # init log files
    log_sim_params = open(LOG_DIR+'/LogSimParams.txt','w+')

    # log simulation params
    text = 'Simulation parameters: \r\n'
    text = text + 'k_rho = %.3f \r\n' % k_rho
    text = text + 'k_alpha = %.3f \r\n' % k_alpha
    text = text + 'k_beta = %.3f \r\n' % k_beta
    text = text + 'v_const = %.3f \r\n' % v_const
    log_sim_params.write(text)
    
    # close log files
    log_sim_params.close()

    print('\r\n' + text)

    # check stability
    stab_dict = { True : 'Satisfied!', False : 'Not Satisfied!'}

    print('Stability Condition: ' + stab_dict[check_stability(k_rho, k_alpha, k_beta)])
    print('Strong Stability Condition: ' + stab_dict[check_strong_stability(k_rho, k_alpha, k_beta)])

    # because of the video recording
    sleep(5)

    while rclpy.ok():
        odomMsg = node.create_subscription(Odometry, '/odom', odom_callback, 10)

        # Get robot posiition and orientation
        (x, y) = getPosition(odomMsg)
        theta = getRotation(odomMsg)

        # Update Trajectory
        X_traj = np.append(X_traj, x)
        Y_traj = np.append(Y_traj, y)
        THETA_traj = np.append(THETA_traj, theta)
        X_goal = np.append(X_goal, X_GOAL)
        Y_goal = np.append(Y_goal, Y_GOAL)
        THETA_goal = np.append(THETA_goal, THETA_GOAL)

        status = robotFeedbackControl(velPub, x, y, theta, X_GOAL, Y_GOAL, radians(THETA_GOAL))

        text = '\r\n'
        text = text + '\r\nx :       %.2f -> %.2f [m]' % (x, X_GOAL)
        text = text + '\r\ny :       %.2f -> %.2f [m]' % (y, Y_GOAL)
        text = text + '\r\ntheta :   %.2f -> %.2f [degrees]' % (degrees(theta), THETA_GOAL)

        if status == 'Goal position reached!':
            # stop the robot
            robotStop(velPub)

            # log trajectory
            np.savetxt(LOG_DIR+'/X_traj.csv', X_traj, delimiter = ' , ')
            np.savetxt(LOG_DIR+'/Y_traj.csv', Y_traj, delimiter = ' , ')
            np.savetxt(LOG_DIR+'/THETA_traj.csv', THETA_traj, delimiter = ' , ')
            np.savetxt(LOG_DIR+'/X_goal.csv', X_goal, delimiter = ' , ')
            np.savetxt(LOG_DIR+'/Y_goal.csv', Y_goal, delimiter = ' , ')
            np.savetxt(LOG_DIR+'/THETA_goal.csv', THETA_goal, delimiter = ' , ')


            rclpy.shutdown('Goal position reached! End of simulation!')
            text = text + '\r\n\r\nGoal position reached! End of simulation!'

        print(text)

        sleep()

    node.destroy_node()
    rclpy.shutdown()

