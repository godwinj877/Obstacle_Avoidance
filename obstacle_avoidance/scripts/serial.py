# !/usr/bin/env python3

import serial
import time

# Open serial connection to Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Change the port as needed
time.sleep(2)  # Allow time for Arduino to reset

# Function to send cmd_vel command
def send_cmd_vel(linear_velocity, angular_velocity):
    cmd_vel = f"cmd_vel: linear={linear_velocity}, angular={angular_velocity}\n"
    ser.write(cmd_vel.encode())

# Example usage
send_cmd_vel(0.5, 0.2)  # Send a cmd_vel command with linear velocity 0.5 and angular velocity 0.2

# Close serial connection
ser.close()