# !/usr/bin/env python3

import numpy as np
import math
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from topic import *
from agent import *

# Global parameters
X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0

GOALS = [[0.0, 1.5]]

MAX_EPISODES = 60
MAX_STEPS_PER_EPISODE = 50

LATERAL_VELOCITY = 0.2
LONGITUDINAL_VELOCITY = 0.2

LOG_FILE_DIR = "/home/godwin/Desktop/BTP/obstacle_avoidance/src/obstacle_avoidance/scripts/DDPG/models/"

LOAD_MODEL = True
MODEL_NAME = LOG_FILE_DIR + "16_05_2024_21_03_12([3.0, 4.0])" #"14_03_2024_10_59_33([3.0, -1.0])" #"15_05_2024_20_40_03([2.0, 1.0])"  "16_05_2024_20_28_30([2.0, 1.0])","

class Mecanum(Node):
    def __init__(self):
        super().__init__('mecanum')
        self.state_dim = 10
        self.action_dim = 8
        self.agent = DDPGAgent(self.state_dim, self.action_dim)
        self.load_model = LOAD_MODEL
        self.model_file = MODEL_NAME
        if LOAD_MODEL:
            self.agent.load_models(MODEL_NAME)

        self.img_cnt = 0
        self.action = 0
        self.x = X_INIT
        self.y = Y_INIT
        self.th = THETA_INIT

        self.vel_pub = VelocityPublisher()
        self.laser_sub = LaserSubscriber()
        self.odom_sub = OdomSubscriber()

        self.noise = OUNoise(self.action_dim)

        self.run()

    def writeText(self, goal):
        self.sim_start_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + f'({goal})'
        path = f'{self.sim_start_time}/'
        path = os.path.join(LOG_FILE_DIR, path)
        log_path = os.path.join(path, 'log/')      
        result_path = os.path.join(path, 'result/')      
        actor_path = os.path.join(path, 'actor/')
        target_actor_path = os.path.join(path, 'target_actor/')
        critic_1_path = os.path.join(path, 'critic_1/')
        critic_2_path = os.path.join(path, 'critic_2/')
        target_critic_1_path = os.path.join(path, 'target_critic_1/')
        target_critic_2_path = os.path.join(path, 'target_critic_2/')
        plots_path = os.path.join(path, 'plots/')
        os.makedirs(log_path)
        os.makedirs(result_path)
        os.makedirs(actor_path)
        os.makedirs(target_actor_path)
        os.makedirs(critic_1_path)
        os.makedirs(critic_2_path)
        os.makedirs(target_critic_1_path)
        os.makedirs(target_critic_2_path)
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
        text = text + 'REWARD CRASHED: -200\r\n'
        text = text + 'REWARD REACHED: 200\r\n'
        self.log_sim_params.write(text)

    def run(self):
        for goal in GOALS:
            self.writeText(goal)
            if self.load_model:
                    self.agent.load_models(self.model_file)
            episode = 1
            episode_reward = []
            self.replay_buffer = ReplayBuffer(buffer_size=40000)
            instance = []
            self.stop_robot()

            while episode <= MAX_EPISODES:
                step = 1
                self.reward = 0
                step_reward = []
                self.stop_robot()

                self.reached = False
                self.crash = False

                while step <= MAX_STEPS_PER_EPISODE:
                    print(f"\n\nEpisode: {episode} | Step: {step} started for goal: {goal}")

                    # State
                    self.collect_data()
                    relative_distance = self.calculate_goal_distance(self.odom_data, goal)
                    relative_heading = self.calculate_heading(self.odom_data, goal)
                    self.current_state = np.concatenate((self.laser_data, [relative_distance, relative_heading]), axis=0)
                    print("State: ", self.current_state)

                    # Checking for crash
                    self.crash = checkCrash(self.laser_data, self.action)
                    if self.crash and step != 1:
                        print('Crashed')
                        self.stop_robot()
                        self.stop_robot()
                        self.stop_robot()
                        self.reward = self.calculate_reward()
                        print("Reward: ", self.reward)
                        step_reward.append(self.reward)
                        self.agent.store_transition(self.current_state, self.action, self.reward, self.next_state, self.crash)
                        self.crash = False
                        self.agent.save_models(self.sim_start_time)
                        break
                    
                    # Checking for goal
                    if relative_distance < 0.1 and step != 1:
                        self.reached = True
                        self.stop_robot()
                        self.get_logger().info("Reached goal!")
                        instance.append([episode, step])
                        self.reward = self.calculate_reward()
                        print("Reward: ", self.reward)
                        step_reward.append(self.reward)
                        self.agent.save_models(self.sim_start_time)
                        self.reached = False
                        break
                    
                    # Executing action with Noise
                    self.action = self.agent.select_action(self.current_state) + self.noise.noise()
                    self.action = np.clip(self.action, -1, 1)
                    self.execute_action()

                    # Next state
                    self.collect_data()
                    next_relative_distance = self.calculate_goal_distance(self.odom_data, goal)
                    next_relative_heading = self.calculate_heading(self.odom_data, goal)

                    self.next_state = np.concatenate((self.laser_data, [next_relative_distance, next_relative_heading]), axis=0)
                    print("Next State: ", self.next_state)

                    # Calculating reward
                    self.reward = self.calculate_reward(relative_distance, next_relative_distance, relative_heading, next_relative_heading)
                    print('Reward: ', self.reward)

                    self.agent.store_transition(self.current_state, self.action, self.reward, self.next_state, self.crash)
                    step_reward.append(self.reward)
                    step = step + 1
                    # sleep(0.2)

                print("Training...")
                for _ in range(50):
                    self.agent.train(self.current_state, self.action, self.reward, self.next_state, self.crash)

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
        rclpy.spin_once(self.laser_sub)
        print("Distance between obstacle: ", np.min(self.laser_sub.laser_data))
        self.laser_data = scanDiscretization(self.laser_sub.laser_data)

    def calculate_goal_distance(self, odom, goal):
        distance = sqrt( pow( ( goal[0] - odom[0] ) , 2 ) + pow( ( goal[1] - odom[1] ) , 2) )
        return round(distance, 2)
    
    def calculate_heading(self, odom, goal):
        dx = goal[0] - odom[0]
        dy = goal[1] - odom[1]

        heading_rad = math.atan2(dy, dx)
        heading_degrees = math.degrees(heading_rad)

        if heading_degrees > 180:
            heading_degrees -= 360
        elif heading_degrees < -180:
            heading_degrees += 360

        return round(heading_degrees)

    def calculate_reward(self, prev_relative_distance=0.0, relative_distance=0.0, prev_relative_heading=0.0, relative_heading=0.0):
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
    
    def execute_action(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = 0.0
        action = np.argmax(self.action)
        print("Action: ", action)
        if action == 0:
            cmd_vel.linear.x = LATERAL_VELOCITY
        elif action == 1:
            cmd_vel.linear.x = -LATERAL_VELOCITY
        elif action == 2:
            cmd_vel.linear.y = LONGITUDINAL_VELOCITY
        elif action == 3:
            cmd_vel.linear.y = -LONGITUDINAL_VELOCITY
        elif action == 4:
            cmd_vel.linear.x = LATERAL_VELOCITY
            cmd_vel.linear.y = LONGITUDINAL_VELOCITY
        elif action == 5:
            cmd_vel.linear.x = -LATERAL_VELOCITY
            cmd_vel.linear.y = LONGITUDINAL_VELOCITY
        elif action == 6:
            cmd_vel.linear.x = -LATERAL_VELOCITY
            cmd_vel.linear.y = -LONGITUDINAL_VELOCITY
        elif action == 7:
            cmd_vel.linear.x = LATERAL_VELOCITY
            cmd_vel.linear.y = -LONGITUDINAL_VELOCITY

        self.vel_pub.publish_velocity(cmd_vel)
        sleep(1)

    def stop_robot(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = 0.0
        self.vel_pub.publish_velocity(cmd_vel)

    def plot(self, episode_reward):
        plt.plot(episode_reward)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Reward vs Episode") 
        plt.grid(True)
        plt.show()
        
        img_path = LOG_FILE_DIR + f'{self.sim_start_time}/plots/img_{self.img_cnt}.jpg'
        plt.savefig(img_path)
        self.img_cnt += 1
        print("Plotted images!!!")
        plt.close()

def main():
    rclpy.init()
    node = Mecanum()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()    

if __name__ == '__main__':
    main()
