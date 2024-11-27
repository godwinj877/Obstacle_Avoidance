import numpy as np
from math import *
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import os
from itertools import *
from topic import *

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
epsilon = 0.5
MAX_EPISODES = 20
MAX_STEPS_PER_EPISODE = 2000

LOG_FILE_DIR = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/SARSA/output/"

X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0

GOALS = [[1.0, 1.0],[3.0, -1.0], [5.0, 0.0]]

class SARSA(Node):
	def __init__(self, learning_rate = ALPHA, discount_factor = GAMMA, epsilon = epsilon):
		super().__init__('sarsa_node')
		self.img_cnt = 0
		self.ALPHA = learning_rate
		self.GAMMA = discount_factor
		self.epsilon = epsilon

		self.lidar_data_space = self.create_lidar_states()
		self.goal_distance_space = range(12)
		self.goal_heading_space = range(360)
		self.action_space = [0,1,2,3,4,5,6,7]

		self.lidar_n = len(self.lidar_data_space)
		self.goal_dist_n = len(self.goal_distance_space)
		self.goal_head_n = len(self.goal_heading_space)
		self.action_n = len(self.action_space)
		
		self.QTable = self.create_QTable()

		self.action = 0
		self.x = X_INIT
		self.y = Y_INIT
		self.th = THETA_INIT

		self.vel_pub = VelocityPublisher()
		self.laser_sub = LaserSubscriber()
		self.odom_sub = OdomSubscriber()

		self.run()

	def create_lidar_states(self):
		x1 = set((0,1,2))
		x2 = set((0,1,2))
		x3 = set((0,1,2))
		x4 = set((0,1,2))
		x5 = set((0,1,2))
		x6 = set((0,1,2))
		x7 = set((0,1,2))
		x8 = set((0,1,2))
		lidar_state = set(product(x1,x2,x3,x4,x5,x6,x7,x8))
		return np.array(list(lidar_state))
		
	def get_state(self, lidar, goal_dis, heading):
		state = [lidar, goal_dis, heading]
		return tuple(state)

	def create_QTable(self):
		QTable = np.random.uniform(low=-2, high=0, size=([self.lidar_n] + [self.goal_dist_n] + [self.goal_head_n] + [self.action_n]))
		return QTable

	def choose_action(self, state):
		# if np.random.rand() <= self.epsilon:
		# 	action = np.random.choice(self.action_space)  
		# else:
		
		action = np.argmax(self.QTable[state])
		if action in self.action_space:
			return action
		else:
			return np.random.randint(0,8)
	
	def update_QTable(self, state, action, reward, next_state, next_action):
		next_q_value = self.QTable[next_state + (next_action,)]
		current_q_value = self.QTable[state + (action,)]
		target_q_value = reward + (self.GAMMA * next_q_value)
		new_q_value = (1 - self.ALPHA) * current_q_value + (self.ALPHA * target_q_value)
		self.QTable[state + (action,)] = new_q_value

	def calculate_goal_distance(self, odom, goal):
		distance = sqrt( pow( ( goal[0] - odom[0] ) , 2 ) + pow( ( goal[1] - odom[1] ) , 2) )
		return round(distance)

	def calculate_heading(self, odom, goal):
		dx = goal[0] - odom[0]
		dy = goal[1] - odom[1]

		heading = atan2(dy, dx)
		heading_degrees = round(degrees(heading))%360
		return heading_degrees

	def calculate_reward(self, prev_relative_distance=0.0, relative_distance=0.0, prev_relative_heading=0.0, relative_heading=0.0):
		print('Calculating Reward')
		r_step = -2
		r_dist = -1
		r_head = -1
		if self.crash:
			r_crash = -200
			return r_crash
		if self.reached:
			r_goal = 200
			return r_goal
		if relative_distance < prev_relative_distance:
			r_dist = 1
		if relative_heading < prev_relative_heading:
			r_head = 1
		reward = r_step + r_dist + r_head
		return reward

	def execute_action(self, goal):
		cmd_vel = Twist()
		cmd_vel.linear.x = 0.0
		cmd_vel.linear.y = 0.0
		cmd_vel.angular.z = 0.0
		action = self.action
		if action == 0:
			cmd_vel.linear.x = 0.25
		elif action == 1:
			cmd_vel.linear.x = -0.25
		elif action == 2:
			cmd_vel.linear.y = 0.25
		elif action == 3:
			cmd_vel.linear.y = -0.25
		elif action == 4:
			cmd_vel.linear.x = 0.25
			cmd_vel.linear.y = 0.25
		elif action == 5:
			cmd_vel.linear.x = -0.25
			cmd_vel.linear.y = 0.25
		elif action == 6:
			cmd_vel.linear.x = -0.25
			cmd_vel.linear.y = -0.25
		elif action == 7:
			cmd_vel.linear.x = 0.25
			cmd_vel.linear.y = -0.25

		self.vel_pub.publish_velocity(cmd_vel)
		sleep(5)

		self.collect_data()
		self.next_laser_data  = self.laser_data
		self.next_odom_data = self.odom_data
		self.next_goal_distance = self.calculate_goal_distance(self.next_odom_data, goal)
		print("Next Goal Distance: ", self.next_goal_distance)
		self.next_heading = self.calculate_heading(self.next_odom_data, goal)
		self.next_state = self.get_state(self.next_laser_data, self.next_goal_distance, self.next_heading)
		print("Next State: ", self.next_state)
		self.reward = self.calculate_reward(self.goal_distance, self.next_goal_distance, self.heading, self.next_heading)
		print('Reward: ', self.reward)

	def stop_robot(self):
		cmd_vel = Twist()
		cmd_vel.linear.x = 0.0
		cmd_vel.linear.y = 0.0
		cmd_vel.angular.z = 0.0
		self.vel_pub.publish_velocity(cmd_vel)

	def writeText(self, goal):
		self.sim_start_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + f'({goal})'
		path = f'{self.sim_start_time}/'
		path = os.path.join(LOG_FILE_DIR, path)
		log_path = os.path.join(path, 'log/')      
		result_path = os.path.join(path, 'result/')      
		plots_path = os.path.join(path, 'plots/')
		os.makedirs(log_path)
		os.makedirs(result_path)
		os.makedirs(plots_path)

		# Open files
		self.log_sim_info = open(log_path+'/LogInfo.txt', 'w+')
		self.log_sim_params = open(log_path+'/LogParams.txt', 'w+')
		
		text = '\r\nSIMULATION START ===> ' + self.sim_start_time + '\r\n'
		self.log_sim_info.write(text)
		self.log_sim_params.write(text)

		# Log simulation params
		text = '\r\nSimulation parameters: \r\n'
		text = text + 'INITIAL POSITION = (%.2f, %.2f, %.2f) \r\n' % (X_INIT, Y_INIT, THETA_INIT)
		text = text + '--------------------------------\r\n'
		text = text + 'MAX EPISODES = %d\r\n' % MAX_EPISODES
		text = text + 'MAX STEPS PER EPISODE = %d\r\n' % MAX_STEPS_PER_EPISODE
		text = text + '--------------------------------\r\n'
		text = text + 'MAX LIDAR DISTANCE = %.2f\r\n' % MAX_LIDAR_DISTANCE
		text = text + 'NEARBY DISTANCE = %.2f\r\n' % NEARBY_DISTANCE
		text = text + 'COLLISION DISTANCE = %.2f\r\n' % COLLISION_DISTANCE
		text = text + 'ZONE 0 LENGTH = %.2f\r\n' % ZONE_0_LENGTH
		text = text + 'ZONE 1 LENGTH = %.2f\r\n' % ZONE_1_LENGTH

		text = text + '--------------------------------\r\n'
		text = text + 'REWARD CRASHED: -10000\r\n'
		text = text + 'REWARD REACHED: 10000\r\n'
		text = text + 'REWARD EACH MOVE: 1000 - RELATIVE_DISTANCE*50\r\n'
		self.log_sim_params.write(text)

	def reset_world(self):
		resetter = GazeboWorldResetter()
		rclpy.spin_once(resetter)
		resetter.destroy_node()
		sleep(5)
 
	def run(self):
		for goal in GOALS:
			episode = 1
			episode_reward = []
			instance = []
			self.reset_world()      
			self.stop_robot()

			while episode <= MAX_EPISODES:
				step = 1
				self.reward = 0
				step_reward = []

				self.reset_world()
				self.stop_robot()
				self.reached = False
				self.crash = False

				while step <= MAX_STEPS_PER_EPISODE:
					print(f"\n\nEpisode: {episode} | Step: {step} started for goal: {goal}")
					self.collect_data()
					self.goal_distance = self.calculate_goal_distance(self.odom_data,goal)
					print("Goal Distance: ", self.goal_distance)
					self.heading = self.calculate_heading(self.odom_data, goal)
					self.current_state = self.get_state(self.laser_data,self.goal_distance, self.heading)
					print("State: ", self.current_state)
					self.crash = checkCrash(self.laser_sub.laser_data, self.action)
					self.reached = checkGoalNear(self.odom_data[0],self.odom_data[1],goal[0],goal[1])

					if self.crash and step != 1:
						print('Crashed')
						self.reward = self.calculate_reward()
						print('Reward: ', self.reward)
						step_reward.append(self.reward)
						self.crash = False
						self.reset_world()
						break
					if self.reached and step != 1:
						print("Reached goal!")
						self.reward = self.calculate_reward()
						step_reward.append(self.reward)
						print('Reward: ', self.reward)
						instance.append([episode, step])
						self.reached = False
						self.reset_world()
						break
					
					self.action = self.choose_action(self.current_state)
					print("Action: ", self.action)
					self.execute_action(goal)
					
					self.next_action = self.choose_action(self.next_state)
					print("Next Action: ", self.next_action)
					self.update_QTable(self.current_state,self.action,self.reached, self.next_state, self.next_action)

					step_reward.append(self.reward)
					step = step + 1
					sleep(2)
				                
				episode_reward.append(np.sum(step_reward))
				episode = episode + 1
				self.stop_robot()
				i = input("Continue!!!")
			
			print("Instances: ", instance)
			self.plot(episode_reward)
			self.stop_robot()

	def collect_data(self):
		rclpy.spin_once(self.odom_sub)
		rclpy.spin_once(self.odom_sub)
		self.odom_data = self.odom_sub.odom_data
		print("Odom: ", self.odom_data)
		
		rclpy.spin_once(self.laser_sub)
		self.laser_data = scanDiscretization(self.laser_sub.laser_data)
		# print("Laser: ", self.laser_data)

	def plot(self, episode_reward):
		plt.plot(episode_reward)
		plt.xlabel("Episode")
		plt.ylabel("Episode Reward")
		plt.title("Reward vs Episode")
		plt.grid(True)
		# plt.show()
		
		img_path = LOG_FILE_DIR + f'{self.sim_start_time}/plots/img_{self.img_cnt}.jpg'
		plt.savefig(img_path)
		self.img_cnt += 1
		print("Plotted images!!!")
		text = '\r\nPlotted and Saved ==> ' + img_path
		text  = text + '\r\n---------------------------------\r\n'
		self.log_sim_info.write(text)
		plt.close()

def main(args=None):
    rclpy.init(args=args)
    node = SARSA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()    

if __name__ == '__main__':
    main()



    