# !/usr/bin/env/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from time import time, sleep
from datetime import datetime
import matplotlib.pyplot as plt
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import sys

DATA_PATH = '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/QLearn_data'
MODULES_PATH = '/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/QLearning'
sys.path.insert(0, MODULES_PATH)

from QLearning import *
from lidar import *
from control import *

# Episode parameters
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 3
MIN_TIME_BETWEEN_ACTIONS = 0

# Learning parameters
ALPHA = 0.5
GAMMA = 0.9

T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001

EPSILON_INIT = 0.9
EPSILON_GRAD = 0.96
EPSILON_MIN = 0.05

# 1 - Softmax , 2 - Epsilon greedy
EXPLORATION_FUNCTION = 2

# Initial position
X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0

RANDOM_INIT_POS = False

# Log file directory
LOG_FILE_DIR = DATA_PATH + '/Log_learning'

# Q table source file
Q_SOURCE_DIR = LOG_FILE_DIR
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

class LearningNode(Node):
    def __init__(self):
        super().__init__('learning_node')
        self.get_logger().info("Learning node initialized")
        self.setPosPub = ModelStatePublisher()
        self.velPub = VelocityPublisher()
        self.odom = PositionSubscriber()
        self.scan = LaserSubscriber()
        self.get_logger().info('SCAN NODE START ==> {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        self.initLearning()
        print("Q Table completed\n")
        self.initParams()
        print("Parameters initialized\n")
        self.Learn()

    def initLearning(self):
        self.actions = createActions()
        self.state_space = createStateSpace()
        if Q_SOURCE_DIR != LOG_FILE_DIR:
            self.Q_table = readQTable(Q_SOURCE_DIR+'/Qtable.csv')
        else:
            self.Q_table = createQTable(len(self.state_space),len(self.actions))
        print('Initial Q-table:')
        print(self.Q_table, len(self.Q_table))

    def initParams(self):
        # Init log files
        self.log_sim_info = open(LOG_FILE_DIR+'/LogInfo.txt','w+')
        self.log_sim_params = open(LOG_FILE_DIR+'/LogParams.txt','w+')

        # Learning parameters
        self.T = T_INIT
        self.EPSILON = EPSILON_INIT
        self.alpha = ALPHA
        self.gamma = GAMMA

        # Episodes, steps, rewards
        self.ep_steps = 0
        self.ep_reward = 0
        self.episode = 1
        self.crash = 0
        self.reward_max_per_episode = np.array([])
        self.reward_min_per_episode = np.array([])
        self.reward_avg_per_episode = np.array([])
        self.ep_reward_arr = np.array([])
        self.steps_per_episode = np.array([])
        self.reward_per_episode = np.array([])

        # Action and state index and Lidar data
        self.action = 0
        self.state_ind = 0
        self.prev_action = 0
        self.prev_lidar = np.zeros(360)
        self.prev_state_ind = 0

        # initial position
        self.robot_in_pos = False
        self.first_action_taken = False

        # init time
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()

        # init timer
        while not (self.t_start.to_msg().nanosec > self.t_0.to_msg().nanosec):
            self.t_start = self.get_clock().now()

        self.t_ep = self.t_start
        self.t_sim_start = self.t_start
        self.t_step = self.t_start

        self.T_per_episode = np.array([])
        self.EPSILON_per_episode = np.array([])
        self.t_per_episode = np.array([])

        # Date
        self.now_start = datetime.now()
        dt_string_start = self.now_start.strftime("%d/%m/%Y %H:%M:%S")

        # Log date to files
        text = '\r\n' + 'SIMULATION START ==> ' + dt_string_start + '\r\n\r\n'
        print(text)
        self.log_sim_info.write(text)
        self.log_sim_params.write(text)

        # Log simulation parameters
        text = '\r\nSimulation parameters: \r\n'
        text = text + '--------------------------------------- \r\n'
        if RANDOM_INIT_POS:
            text = text + 'INITIAL POSITION = RANDOM \r\n'
        else:
            text = text + 'INITIAL POSITION = ( %.2f , %.2f , %.2f ) \r\n' % (X_INIT,Y_INIT,THETA_INIT)
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_EPISODES = %d \r\n' % MAX_EPISODES
        text = text + 'MAX_STEPS_PER_EPISODE = %d \r\n' % MAX_STEPS_PER_EPISODE
        text = text + 'MIN_TIME_BETWEEN_ACTIONS = %.2f s \r\n' % MIN_TIME_BETWEEN_ACTIONS
        text = text + '--------------------------------------- \r\n'
        text = text + 'ALPHA = %.2f \r\n' % ALPHA
        text = text + 'GAMMA = %.2f \r\n' % GAMMA

        if EXPLORATION_FUNCTION == 1:
            text = text + 'T_INIT = %.3f \r\n' % T_INIT
            text = text + 'T_GRAD = %.3f \r\n' % T_GRAD
            text = text + 'T_MIN = %.3f \r\n' % T_MIN
        else:
            text = text + 'EPSILON_INIT = %.3f \r\n' % EPSILON_INIT
            text = text + 'EPSILON_GRAD = %.3f \r\n' % EPSILON_GRAD
            text = text + 'EPSILON_MIN = %.3f \r\n' % EPSILON_MIN
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_LIDAR_DISTANCE = %.2f \r\n' % MAX_LIDAR_DISTANCE
        text = text + 'COLLISION_DISTANCE = %.2f \r\n' % COLLISION_DISTANCE
        text = text + 'ZONE_0_LENGTH = %.2f \r\n' % ZONE_0_LENGTH
        text = text + 'ZONE_1_LENGTH = %.2f \r\n' % ZONE_1_LENGTH
        text = text + '--------------------------------------- \r\n'
        text = text + 'CONST_LATERAL_SPEED_FORWARD = %.3f \r\n' % CONST_LATERAL_SPEED_FORWARD
        text = text + 'CONST_LONGITUDINAL_SPEED_FORWARD = %.3f \r\n' % CONST_LONGITUDINAL_SPEED_FORWARD
        text = text + 'CONST_ANGULAR_SPEED_FORWARD = %.3f \r\n' % CONST_ANGULAR_SPEED_FORWARD
        text = text + 'CONST_LATERAL_SPEED_LEFT = %.3f \r\n' % CONST_LATERAL_SPEED_LEFT
        text = text + 'CONST_LONGITUDINAL_SPEED_LEFT = %.3f \r\n' % CONST_LONGITUDINAL_SPEED_LEFT
        text = text + 'CONST_ANGULAR_SPEED_LEFT = %.3f \r\n' % CONST_ANGULAR_SPEED_LEFT
        text = text + 'CONST_LATERAL_SPEED_TURN = %.3f \r\n' % CONST_LATERAL_SPEED_TURN
        text = text + 'CONST_LONGITUDINAL_SPEED_TURN = %.3f \r\n' % CONST_LONGITUDINAL_SPEED_TURN
        text = text + 'CONST_ANGULAR_SPEED_TURN = %.3f \r\n' % CONST_ANGULAR_SPEED_TURN
        self.log_sim_params.write(text)

    def Learn(self):
        
        #self.state_space = createStateSpace()     
        print("After creating: ", self.state_space)
        print("\n")
        while rclpy.ok:
            print("\n\n")
            print("Entered loop")
            step_time = (self.get_clock().now() - self.t_step).to_msg().sec
            self.t_step = self.get_clock().now()
            if step_time > 100:
                print("Entered big step time")
                text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
                print(text)
                self.log_sim_info.write(text+'\r\n')

            # End of Learning
            if self.episode > MAX_EPISODES:
                print("Reached Maximun number of episodes")
                sim_time = (self.get_clock().now() - self.t_sim_start).to_msg().sec
                sim_time_h = sim_time // 3600
                sim_time_m = ( sim_time - sim_time_h * 3600 ) // 60
                sim_time_s = sim_time - sim_time_h * 3600 - sim_time_m * 60

                # real time
                self.now_stop = datetime.now()
                dt_string_stop = self.now_stop.strftime("%d/%m/%Y %H:%M:%S")
                real_time_delta = (self.now_stop - self.now_start).total_seconds()
                real_time_h = real_time_delta // 3600
                real_time_m = ( real_time_delta - real_time_h * 3600 ) // 60
                real_time_s = real_time_delta - real_time_h * 3600 - real_time_m * 60

                # Log learning session info to file
                text = '--------------------------------------- \r\n\r\n'
                text = text + 'MAX EPISODES REACHED(%d), LEARNING FINISHED ==> ' % MAX_EPISODES + dt_string_stop + '\r\n'
                text = text + 'Simulation time: %d:%d:%d  h/m/s \r\n' % (sim_time_h, sim_time_m, sim_time_s)
                text = text + 'Real time: %d:%d:%d  h/m/s \r\n' % (real_time_h, real_time_m, real_time_s)
                print(text)
                self.log_sim_info.write('\r\n'+text+'\r\n')
                self.log_sim_params.write(text+'\r\n')

                # Log data to file
                saveQTable(LOG_FILE_DIR+'/Qtable.csv', self.Q_table)
                np.savetxt(LOG_FILE_DIR+'/StateSpace.csv', self.state_space, '%d')
                np.savetxt(LOG_FILE_DIR+'/steps_per_episode.csv', self.steps_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_per_episode.csv', self.reward_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/T_per_episode.csv', self.T_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/EPSILON_per_episode.csv', self.EPSILON_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_min_per_episode.csv', self.reward_min_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_max_per_episode.csv', self.reward_max_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_avg_per_episode.csv', self.reward_avg_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/t_per_episode.csv', self.t_per_episode, delimiter = ' , ')

                # Close files and shut down node
                self.log_sim_info.close()
                self.log_sim_params.close()
                rclpy.shutdown()
                break
            else:
                print("Started Learning\n")
                self.ep_time = (self.get_clock().now() - self.t_ep).to_msg().sec
                print("Is crashed?: ", self.crash)
                if self.crash or self.ep_steps >= MAX_STEPS_PER_EPISODE:
                    robotStop(self.velPub)

                    if self.crash:
                        print("Crashed")
                        #rclpy.spin_once(self.odom)
                        (self.x_crash, self.y_crash) = getPosition(self.odom)
                        self.theta_crash = degrees(getRotation(self.odom))

                    self.t_ep = self.get_clock().now()
                    self.reward_min = np.min(self.ep_reward_arr)
                    self.reward_max = np.max(self.ep_reward_arr)
                    self.reward_avg = np.mean(self.ep_reward_arr)
                    now = datetime.now()
                    dt_string = now.strftime("%d%m%Y %H:%M:%S")

                    text = '---------------------------------------\r\n'
                    if self.crash:
                        text = text + '\r\nEpisode %d ==> CRASH {%.2f,%.2f,%.2f}    ' % (self.episode, self.x_crash, self.y_crash, self.theta_crash) + dt_string
                    elif self.ep_steps >= MAX_STEPS_PER_EPISODE:
                        text = text + '\r\nEpisode %d ==> MAX STEPS PER EPISODE REACHED {%d}    ' % (self.episode, MAX_STEPS_PER_EPISODE) + dt_string
                    else:
                        text = text + '\r\nEpisode %d ==> UNKNOWN TERMINAL CASE    ' % self.episode + dt_string
                    text = text + '\r\nepisode time: %.2f s (avg step: %.2f s) \r\n' % (self.ep_time, self.ep_time / self.ep_steps)
                    text = text + 'episode steps: %d \r\n' % self.ep_steps
                    text = text + 'episode reward: %.2f \r\n' % self.ep_reward
                    text = text + 'episode max | avg | min reward: %.2f | %.2f | %.2f \r\n' % (self.reward_max, self.reward_avg, self.reward_min)
                    if EXPLORATION_FUNCTION == 1:
                        text = text + 'T = %f \r\n' % self.T
                    else:
                        text = text + 'EPSILON = %f \r\n' % self.EPSILON
                    print(text)
                    self.log_sim_info.write('\r\n'+text)

                    self.steps_per_episode = np.append(self.steps_per_episode, self.ep_steps)
                    self.reward_per_episode = np.append(self.reward_per_episode, self.ep_reward)
                    self.T_per_episode = np.append(self.T_per_episode, self.T)
                    self.EPSILON_per_episode = np.append(self.EPSILON_per_episode, self.EPSILON)
                    self.t_per_episode = np.append(self.t_per_episode, self.ep_time)
                    self.reward_min_per_episode = np.append(self.reward_min_per_episode, self.reward_min)
                    self.reward_max_per_episode = np.append(self.reward_max_per_episode, self.reward_max)
                    self.reward_avg_per_episode = np.append(self.reward_avg_per_episode, self.reward_avg)
                    self.ep_reward_arr = np.array([])
                    self.ep_steps = 0
                    self.ep_reward = 0
                    self.crash = False
                    self.robot_in_pos = False
                    self.first_action_taken = False
                    if self.T > T_MIN:
                        self.T = T_GRAD * self.T
                    if self.EPSILON > EPSILON_MIN:
                        self.EPSILON = EPSILON_GRAD * self.EPSILON
                    self.episode = self.episode + 1
                    resetter = GazeboWorldResetter()
                    rclpy.spin_once(resetter)
                    resetter.destroy_node()
                else:
                    self.ep_steps = self.ep_steps + 1
                    print("Checking for action")
                    print("Is robot in position?: ", self.robot_in_pos)
                    print("Is first action?: ", self.first_action_taken)
                    # Initial position
                    while not self.robot_in_pos:
                        print("Initializing position")
                        robotStop(self.velPub)
                        self.ep_steps = self.ep_steps - 1
                        self.first_action_taken = False

                        # init pos
                        if RANDOM_INIT_POS:
                            ( self.x_init , self.y_init , self.theta_init ) = robotSetRandomPose(self.setPosPub)
                        else:
                            ( self.x_init , self.y_init , self.theta_init ) = robotSetPose(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)
                        print("x, y, theta initial: ", self.x_init, self.y_init, self.theta_init)
                        
                        #rclpy.spin_once(self.odom)
                        (self.x, self.y) = getPosition(self.odom)
                        self.theta = degrees(getRotation(self.odom))
                        print("x, y, theta: ", self.x, self.y, self.theta)

                        # check init pos
                        if abs(self.x-self.x_init) < 0.01 and abs(self.y-self.y_init) < 0.01 and abs(self.theta-self.theta_init) < 1:
                            self.robot_in_pos = True
                            print("Robot in position: ", self.robot_in_pos)
                            print("Robot first action: ", self.first_action_taken)
                        else:
                            self.robot_in_pos = False 
                            print("Robot not in position")

                    # First acion
                    while not self.first_action_taken:
                        print("First action")
                        self.scan = LaserSubscriber()
                        rclpy.spin_once(self.scan)
                        ( lidar, angles ) = lidarScan(self.scan)
                        ( state_ind, x1, x2 ,x3 ,x4, x5, x6, x7, x8 ) = scanDiscretization(self.state_space, lidar)
                        self.crash = checkCrash(lidar, self.prev_action)
                        print("Is crashed?: ", self.crash)
                        
                        if EXPLORATION_FUNCTION == 1 :
                            ( action, status_strat ) = softMaxSelection(self.Q_table, self.state_ind, self.actions, self.T)
                        else:
                            ( action, status_strat ) = epsilonGreedyExploration(self.Q_table, self.state_ind, self.actions, self.T)
                        
                        print("Action : ", action)
                        status_rda = robotDoAction(self.velPub, action)

                        self.prev_lidar = lidar
                        self.prev_action = action
                        self.prev_state_ind = state_ind

                        self.first_action_taken = True

                        if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                            print('\r\n', status_strat, '\r\n')
                            self.log_sim_info.write('\r\n'+status_strat+'\r\n')

                        if not status_rda == 'robotDoAction => OK':
                            print('\r\n', status_rda, '\r\n')
                            self.log_sim_info.write('\r\n'+status_rda+'\r\n')
                        
                    # Rest of the algorithm
                    print("Started learning")
                    rclpy.spin_once(self.scan)
                    ( lidar, angles ) = lidarScan(self.scan)
                    ( state_ind, x1, x2 ,x3 ,x4, x5, x6, x7, x8 ) = scanDiscretization(self.state_space, lidar)
                    self.crash = checkCrash(lidar, self.prev_action)
                    print("Is crashed?: ", self.crash)

                    ( reward, terminal_state ) = getReward(self.action, self.prev_action, lidar, self.prev_lidar, self.crash)

                    ( self.Q_table, status_uqt ) = updateQTable(self.Q_table, self.prev_state_ind, self.action, reward, self.state_ind, self.alpha, self.gamma)
                    
                    print("\n\nQ Table: ")
                    print(self.Q_table)
                    print("\n\n")
            
                    if EXPLORATION_FUNCTION == 1:
                        ( action, status_strat ) = softMaxSelection(self.Q_table, self.state_ind, self.actions, self.T)
                    else:
                        ( action, status_strat ) = epsilonGreedyExploration(self.Q_table, self.state_ind, self.actions, self.T)
                    
                    print("Action: ", action)
                    status_rda = robotDoAction(self.velPub, action)

                    if not status_uqt == 'updateQTable => OK':
                        print('\r\n', status_uqt, '\r\n')
                        self.log_sim_info.write('\r\n'+status_uqt+'\r\n')
                    if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                        print('\r\n', status_strat, '\r\n')
                        self.log_sim_info.write('\r\n'+status_strat+'\r\n')
                    if not status_rda == 'robotDoAction => OK':
                        print('\r\n', status_rda, '\r\n')
                        self.log_sim_info.write('\r\n'+status_rda+'\r\n')

                    self.ep_reward = self.ep_reward + reward
                    self.ep_reward_arr = np.append(self.ep_reward_arr, reward)
                    self.prev_lidar = lidar
                    self.prev_action = action
                    self.prev_state_ind = state_ind

                    print("Episode: ", self.episode)
                    print("Step: ", self.ep_steps)
                    print("Episode Reward: ", self.ep_reward)
                    sleep(1)

if __name__ == '__main__':
    rclpy.init()
    learning_node = LearningNode()
    rclpy.spin(learning_node)
    learning_node.destroy_node()
    rclpy.shutdown()
