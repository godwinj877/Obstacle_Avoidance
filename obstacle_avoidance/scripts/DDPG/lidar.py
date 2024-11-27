# !/usr/bin/env/python3

import numpy as np
from math import *

MAX_LIDAR_DISTANCE = 5.0
COLLISION_DISTANCE = 1.0
NEARBY_DISTANCE = 2.0

ZONE_0_LENGTH = 1.0
ZONE_1_LENGTH = 2.0

ANGLE_MAX = 359
ANGLE_MIN = 0
ANGLE_MID = 180
HORIZON_WIDTH = 45

# Convert LasecScan msg to array
def lidarScan(msgScan):
    distances = np.array([])
    angles = np.array([])

    for i in range(len(msgScan.ranges)):
        angle = degrees(i * msgScan.angle_increment)
        if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
            distance = MAX_LIDAR_DISTANCE
        elif ( msgScan.ranges[i] < msgScan.range_min ):
            distance = msgScan.range_min
            # For real robot - protection
            if msgScan.ranges[i] < 0.05:
                distance = MAX_LIDAR_DISTANCE
                0
        else:
            distance = msgScan.ranges[i]

        distances = np.append(distances, distance)
        angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
    return ( distances, angles )

# Discretization of lidar scan
def scanDiscretization(lidar):
    x1 = 2 # Front Left zone (no obstacle detected)
    x2 = 2 # Front Right zone (no obstacle detected)

    x3 = 2 # Left Left zone (no obstacle detected)
    x4 = 2 # Left Right zone (no obstacle detected)

    x5 = 2 # Back Left zone (no obstacle detected)
    x6 = 2 # Back Right zone (no obstacle detected)
    
    x7 = 2 # Right Left zone (no obstacle detected)
    x8 = 2 # Right Right zonej (no obstacle detected)

    # Find the front left side lidar values of the vehicle
    lidar_front_left = np.min(lidar[(0):(89)])

    if ZONE_1_LENGTH > lidar_front_left and lidar_front_left > ZONE_0_LENGTH:
        x6 = 1 # zone 1 (x1)
    elif lidar_front_left <= ZONE_0_LENGTH:
        x6 = 0 # zone 0 (x1)

    # Find the front right side lidar values of the vehicle
    lidar_front_right = np.min(lidar[(630):(719)])
    if ZONE_1_LENGTH > lidar_front_right and lidar_front_right > ZONE_0_LENGTH:
        x5 = 1 # zone 1 (x2)
    elif lidar_front_right <= ZONE_0_LENGTH:
        x5 = 0 # zone 0 (x2)
    
    # Find the left left side lidar values of the vehicle
    lidar_left_left = np.min(lidar[(180):(269)])
    if ZONE_1_LENGTH > lidar_left_left and lidar_left_left > ZONE_0_LENGTH:
        x7 = 1 # zone 1 (x3)
    elif lidar_left_left <= ZONE_0_LENGTH:
        x7 = 0 # zone 0 (x3)

    # Find the left left side lidar values of the vehicle
    lidar_left_right = np.min(lidar[(90):(179)])
    if ZONE_1_LENGTH > lidar_left_right and lidar_left_right > ZONE_0_LENGTH:
        x8 = 1 # zone 1 (x4)
    elif lidar_left_right <= ZONE_0_LENGTH:
        x8 = 0 # zone 0 (x4)

    # Find the back left side lidar values of the vehicle
    lidar_back_left = np.min(lidar[(270):(359)])
    if ZONE_1_LENGTH > lidar_back_left and lidar_back_left > ZONE_0_LENGTH:
        x2 = 1 # zone 1 (x5)
    elif lidar_back_left <= ZONE_0_LENGTH:
        x2 = 0 # zone 0 (x5)

    # Find the back right side lidar values of the vehicle
    lidar_back_right = np.min(lidar[(360):(449)])
    if ZONE_1_LENGTH > lidar_back_right and lidar_back_right > ZONE_0_LENGTH:
        x1 = 1 # zone 1 (x6)
    elif lidar_back_right <= ZONE_0_LENGTH:
        x1 = 0 # zone 0 (x6)

    # Find the back left side lidar values of the vehicle
    lidar_right_left = np.min(lidar[(540):(629)])
    if ZONE_1_LENGTH > lidar_right_left and lidar_right_left > ZONE_0_LENGTH:
        x3 = 1 # zone 1 (x7)
    elif lidar_right_left <= ZONE_0_LENGTH:
        x3 = 0 # zone 0 (x7)

    # Find the back right side lidar values of the vehicle
    lidar_right_right = np.min(lidar[(450):(539)])
    if ZONE_1_LENGTH > lidar_right_right and lidar_right_right > ZONE_0_LENGTH:
        x4 = 1 # zone 1 (x8)
    elif lidar_right_right <= ZONE_0_LENGTH:
        x4 = 0 # zone 0 (x8)

    return (x1, x2, x3 , x4 , x5 , x6, x7 , x8)


# Check - crash
def checkCrash(lidar, prev_action):
    prev_action = np.argmax(prev_action)
    if prev_action == 0 and (lidar[0] == 0 or lidar[1] == 0):
        return True
    elif prev_action == 1 and (lidar[4] == 0 or lidar[5] == 0):
        return True
    elif prev_action == 2 and (lidar[2] == 0 or lidar[3] == 0):
        return True
    elif prev_action == 3 and (lidar[6] == 0 or lidar[7] == 0):
        return True
    elif prev_action == 4 and (lidar[0] == 0 or lidar[3] == 0):
        return True
    elif prev_action == 5 and (lidar[2] == 0 or lidar[4] == 0):
        return True
    elif prev_action == 6 and (lidar[5] == 0 or lidar[7] == 0):
        return True
    elif prev_action == 7 and (lidar[1] == 0 or lidar[6] == 0):
        return True
    return False
    
    
    
# Check - object nearby
def checkObjectNearby(lidar, prev_action):
    prev_action = np.argmax(prev_action)
    if prev_action == 0:
        lidar_horizon =   lidar[ANGLE_MIN - HORIZON_WIDTH:ANGLE_MAX + HORIZON_WIDTH] 
    elif prev_action == 1:
        lidar_horizon =   lidar[ANGLE_MID - HORIZON_WIDTH:ANGLE_MID + HORIZON_WIDTH] 
    elif prev_action == 2:
        lidar_horizon =   lidar[ANGLE_MIN + HORIZON_WIDTH:ANGLE_MID - HORIZON_WIDTH] 
    elif prev_action == 3:
        lidar_horizon = lidar[ANGLE_MID + HORIZON_WIDTH: ANGLE_MAX - HORIZON_WIDTH]
    elif prev_action == 4:
        lidar_horizon =   lidar[ANGLE_MIN:ANGLE_MIN + 2*HORIZON_WIDTH]
    elif prev_action == 5:
        lidar_horizon =   lidar[ANGLE_MID - 2*HORIZON_WIDTH:ANGLE_MID] 
    elif prev_action == 6:
        lidar_horizon =   lidar[ANGLE_MID :ANGLE_MID + 2*HORIZON_WIDTH] 
    elif prev_action == 7:
        lidar_horizon =   lidar[ANGLE_MAX - 2*HORIZON_WIDTH: ANGLE_MAX]
    print("\n\n")
    print("Lidar Horizon: ", lidar_horizon)
    print("Distance close to an object: ", np.min(lidar_horizon))
    if np.min( lidar_horizon ) < NEARBY_DISTANCE:
        return True
    else:
        return False
    
# Check - goal near
def checkGoalNear(x, y, x_goal, y_goal):
    return abs(x - x_goal) < 0.3 and abs(y - y_goal) < 0.3
    distance = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
    if distance < 0.3:
        return True
    else:
        return False