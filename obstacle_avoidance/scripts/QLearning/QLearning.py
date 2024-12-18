# !/usr/bin/env/python3

import numpy as np
from math import *
from std_msgs.msg import String
from itertools import product
from sensor_msgs.msg import LaserScan

STATE_SPACE_IND_MAX = 20736 - 1
STATE_SPACE_IND_MIN = 1 - 1
ACTIONS_IND_MAX = 5
ACTIONS_IND_MIN = 0

ANGLE_MAX = int(360 - 1)
ANGLE_MIN = int(1 - 1)
ANGLE_MID = int((ANGLE_MIN + ANGLE_MAX + 1)//2)
HORIZON_WIDTH = 75

T_MIN = 0.001

# Create actions
def createActions():
    actions = np.array([0,1,2,3,4,5])
    return actions

# Create state space for Q table
def createStateSpace():
    x1 = set((0,1,2))
    x2 = set((0,1,2))
    x3 = set((0,1,2,3))
    x4 = set((0,1,2,3))
    x5 = set((0,1,2))
    x6 = set((0,1,2))
    x7 = set((0,1,2,3))
    x8 = set((0,1,2,3))
    state_space = set(product(x1,x2,x3,x4,x5,x6,x7,x8)) 
    return np.array(list(state_space))

# Create Q table, dim: n_states x n_actions
def createQTable(n_states, n_actions):
    #Q_table = np.random.uniform(low = -0.05, high = 0, size = (n_states,n_actions) )
    Q_table = np.zeros((n_states, n_actions))
    return Q_table

# Read Q table from path
def readQTable(path):
    Q_table = np.genfromtxt(path, delimiter = ' , ')
    return Q_table

# Write Q table to path
def saveQTable(path, Q_table):
    np.savetxt(path, Q_table, delimiter = ' , ')

# Select the best action a in state
def getBestAction(Q_table, state_ind, actions):
    if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
        status = 'getBestAction => OK'
        a_ind = np.argmax(Q_table[state_ind,:])
        a = actions[a_ind]
    else:
        status = 'getBestAction => INVALID STATE INDEX'
        a = getRandomAction(actions)

    return ( a, status )


# Select random action from actions
def getRandomAction(actions):
    n_actions = len(actions)
    a_ind = np.random.randint(n_actions)
    return actions[a_ind]

# Epsilon Greedy Exploration action chose
def epsilonGreedyExploration(Q_table, state_ind, actions, epsilon):
    if np.random.uniform() > epsilon and STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
        status = 'epsiloGreedyExploration => OK'
        ( a, status_gba ) = getBestAction(Q_table, state_ind, actions)
        if status_gba == 'getBestAction => INVALID STATE INDEX':
            status = 'epsiloGreedyExploration => INVALID STATE INDEX'
    else:
        status = 'epsiloGreedyExploration => OK'
        a = getRandomAction(actions)

    return ( a, status )

# SoftMax Selection
def softMaxSelection(Q_table, state_ind, actions, T):
    if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
        status = 'softMaxSelection => OK'
        n_actions = len(actions)
        P = np.zeros(n_actions) 

        # Boltzman distribution
        P = np.exp(Q_table[state_ind,:] / T)
        P /= np.sum(np.exp(Q_table[state_ind,:] / T))

        if T < T_MIN or np.any(np.isnan(P)):
            ( a, status_gba ) = getBestAction(Q_table, state_ind, actions)
            if status_gba == 'getBestAction => INVALID STATE INDEX':
                status = 'softMaxSelection => INVALID STATE INDEX'
        else:
            rand = np.random.uniform()
            status = 'softMaxSelection => OK'
            if P[0] > rand:
                a = 0
            elif P[0] <= rand and (P[0] + P[1]) > rand:
                a = 1
            elif (P[0] + P[1]) <= rand and (P[0] + P[1] + P[2]) > rand:
                a = 2
            elif (P[0] + P[1] + P[2]) <= rand and (P[0] + P[1] + P[2] + P[3]) > rand:
                a = 3
            elif (P[0] + P[1] + P[2] + P[3]) <= rand and (P[0] + P[1] + P[2] + P[3] + P[4]) > rand:
                a = 4
            elif (P[0] + P[1] + P[2] + P[3] + P[4]) <= rand:
                a = 5
            else:
                status = 'softMaxSelection => Boltzman distribution error => getBestAction '
                status = status + '\r\nP = (%f , %f , %f, %f, %f, %f) , rnd = %f' % (P[0],P[1],P[2],P[3],P[4],P[5],rand)
                status = status + '\r\nQ(%d,:) = ( %f, %f, %f, %f, %f, %f) ' % (state_ind,Q_table[state_ind,0],Q_table[state_ind,1],Q_table[state_ind,2],Q_table[state_ind,3],Q_table[state_ind,4],Q_table[state_ind,5])
                ( a, status_gba ) = getBestAction(Q_table, state_ind, actions)
                if status_gba == 'getBestAction => INVALID STATE INDEX':
                    status = 'softMaxSelection => INVALID STATE INDEX'
    else:
        status = 'softMaxSelection => INVALID STATE INDEX'
        a = getRandomAction(actions)

    return ( a, status )


# Reward function for Q-learning - table
def getReward(action, prev_action, lidar, prev_lidar, crash):
    if crash:
        terminal_state = True
        reward = -500
    else:
        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1],lidar[(ANGLE_MID-HORIZON_WIDTH):(ANGLE_MID):-1], lidar[(ANGLE_MID):(ANGLE_MID+HORIZON_WIDTH):-1]))
        prev_lidar_horizon = np.concatenate((prev_lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],prev_lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1],prev_lidar[(ANGLE_MID-HORIZON_WIDTH):(ANGLE_MID):-1], prev_lidar[(ANGLE_MID):(ANGLE_MID+HORIZON_WIDTH):-1]))
        terminal_state = False
        
        if action == 0 or action == 1 or action == 2 or action == 3:
            r_action = +1.0
        else:
            r_action = -0.5
        # Reward from crash distance to obstacle change
        #W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
        #W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))
        if np.sum(( lidar_horizon - prev_lidar_horizon) ) >= 0:
            r_obstacle = np.sum((lidar_horizon - prev_lidar_horizon))*5
        else:
            r_obstacle = -np.sum((lidar_horizon - prev_lidar_horizon))*5
        # Reward from turn left/right change
        if ((prev_action == 0 and action == 1) or (prev_action == 1 and action == 0) or 
            (prev_action == 2 and action == 3) or (prev_action == 3 and action == 2) or
             (prev_action == 4 and action == 5) or (prev_action == 5 and action == 4)):
            r_change = -4.0
        else:
            r_change = 0.0

        # Cumulative reward
        reward = r_action + r_obstacle + r_change

    return ( reward, terminal_state )


# Update Q-table values
def updateQTable(Q_table, state_ind, action, reward, next_state_ind, alpha, gamma):
    if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX and STATE_SPACE_IND_MIN <= next_state_ind <= STATE_SPACE_IND_MAX:
        status = 'updateQTable => OK'
        print("Reward: ", reward)
        print("State Index: ", state_ind)
        Q_table[state_ind,action] = ( 1 - alpha ) * Q_table[state_ind,action] + alpha * ( reward + gamma * max(Q_table[next_state_ind,:]) )
        print("Q_table[state_ind,action]: ", Q_table[state_ind,action])
    else:
        status = 'updateQTable => INVALID STATE INDEX'
    return ( Q_table, status )
