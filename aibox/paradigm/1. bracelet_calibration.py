# System
import sys
import os

# Use the project file packages instead of the conda packages, i.e. add to system path for import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import keyboard
import time
import json

from pybelt.belt_controller import (BeltOrientationType, BeltVibrationPattern)

from bracelet import connect_belt
from controller import close_app

def calibrate_intensity(direction):
    
    intensity = 5 # initial value

    orientation_mapping = {"bottom": 60,
                           "top": 90,
                           "left": 120,
                           "right": 45}
    
    orientation = orientation_mapping[direction]

    while True:
        if belt_controller:
            belt_controller.send_vibration_command(
                channel_index=0,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=intensity,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=orientation,  # down
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
        print(f'Vibrating at intensity {intensity}.')

        if keyboard.is_pressed('up') and intensity < 100:
            intensity += 5
            time.sleep(0.1)
        elif keyboard.is_pressed('down') and intensity > 5: # no reason to vibrate with intensity of 0
            intensity -= 5
            time.sleep(0.1)
        elif keyboard.is_pressed('y'):
            belt_controller.stop_vibration()
            time.sleep(1)
            return intensity

if __name__ == '__main__':

    participant = 1
    output_path = str(parent_dir) + '/results/'

    connection_check, belt_controller = connect_belt()
    if connection_check:
        print('Bracelet connection successful.')
    else:
        print('Error connecting bracelet. Aborting.')
        sys.exit()

    directions = ["bottom", "top", "left", "right"]
    output = {}

    try:
        for motor_direction in directions:
            motor_intensity = calibrate_intensity(motor_direction)
            print(f"Direction: {motor_direction}, intensity: {motor_intensity}")
            output[motor_direction] = motor_intensity

        with open(output_path + f"calibration_participant_{participant}.json", "w") as json_file:
            json.dump(output, json_file)

    except KeyboardInterrupt:
        close_app(belt_controller)
    
    # In the end, kill everything
    close_app(belt_controller)