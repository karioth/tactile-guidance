# region Setup

import time
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)
from auto_connect import interactive_belt_connect, setup_logger
import threading
import sys
from pynput.keyboard import Key, Listener
import numpy as np
from queue import PriorityQueue
import cv2

# Depth navigation functions

from resources.depth_navigation_functions import map_obstacles, astar, smooth_path, check_obstacles_between_points


# endregion

class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass


def connect_belt():
    setup_logger()

    belt_controller_delegate = Delegate()
    belt_controller = BeltController(belt_controller_delegate)

    # Interactive script to connect the belt
    interactive_belt_connect(belt_controller)

    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        return False, belt_controller
    else:
        # Change belt mode to APP mode
        belt_controller.set_belt_mode(BeltMode.APP_MODE)
        return True, belt_controller


def choose_detection(bboxes, previous_bbox=None, w=1920, h=1080):
    # Hyperparameters
    track_id_weight = 1000
    exponential_weight = 2
    distance_weight = 100

    #print(f'\nPrevious BB: {previous_bbox}')

    candidates = []
    for bbox in bboxes: # x, y, w, h, id, cls, conf
        # bbox has to be within image dimensions
        if bbox[0] <= w and bbox[1] <= h:
            # confidence score
            confidence = bbox[6] # in [0,1]
            confidence_score = exponential_weight**confidence - 1 # exponential growth in [0,1], could also use np.exp() and normalize
            # tracking score
            current_track_id = bbox[4]
            previous_track_id = previous_bbox[4] if previous_bbox is not None else -1
            track_id_score = track_id_weight if current_track_id == previous_track_id else 1 # 1|ꝏ
            # distance score
            if previous_bbox is None:
                distance = None
                distance_inverted = 1
            else:
                current_location = bbox[:2]
                previous_location = previous_bbox[:2]
                distance = np.linalg.norm(current_location - previous_location)
                distance_inverted = 1 / distance if distance >= 1 else distance_weight

            # total score
            score = track_id_score * confidence_score * distance_inverted
            #print(f'Current BB: {bbox}')
            #print(f'TrackID = {current_track_id}, confidence = {confidence}, distance = {distance}')
            #print(f'Score {score} = {track_id_score} * {confidence_score} * {distance_inverted}\n')

            # Possible scores:
            # ꝏ -- same trackingID
            # 100 -- different trackingID, matching BBs (max. 1px deviation), conf=1
            # [0,1] -- different trackingID, BBs distance in [1., sqrt(w^2*h^2)], conf=1
            candidates.append(score)
        else:
            candidates.append(0)

    true_detection = bboxes[np.argmax(candidates)] if len(candidates) else None

    return true_detection


def calibrate_intensity():
    # to be implemented
    return 50


def get_intensity(handBB, targetBB, vibration_intensities, depth_img):

    # Calculate angle
    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]
    angle_radians = np.arctan2(yc_hand - yc_target, xc_target - xc_hand) # inverted y-axis
    angle = np.degrees(angle_radians) % 360
    print(angle)

    # Initialize motor intensities
    right_intensity = 0
    left_intensity = 0
    top_intensity = 0
    bottom_intensity = 0

    max_bottom_intensity, max_top_intensity, max_left_intensity, max_right_intensity = vibration_intensities["bottom"], vibration_intensities["top"], vibration_intensities["left"], vibration_intensities["right"]

    # Calculate motor intensities based on the angle
    if 0 <= angle < 90:
        right_intensity = (90 - angle) / 90 * max_right_intensity
        top_intensity = angle / 90 * max_top_intensity
    elif 90 <= angle < 180:
        top_intensity = (180 - angle) / 90 * max_top_intensity
        left_intensity = (angle - 90) / 90 * max_left_intensity
    elif 180 <= angle < 270:
        left_intensity = (270 - angle) / 90 * max_left_intensity
        bottom_intensity = (angle - 180) / 90 * max_bottom_intensity
    elif 270 <= angle < 360:
        bottom_intensity = (360 - angle) / 90 * max_bottom_intensity
        right_intensity = (angle - 270) / 90 * max_right_intensity

    if type(depth_img) != None:
        
        return int(right_intensity), int(left_intensity), int(top_intensity), int(bottom_intensity), 50

    # front / back motor (depth), currently it is used for grasping signal until front motor is added
    # If there is an anything between hand and target that can be hit (depth smaller than depth of both target and image) - move backwards

    roi_x_min, roi_x_max, roi_y_min, roi_y_max = int(min(xc_target, xc_hand)), int(max(xc_target, xc_hand)), int(min(yc_target, yc_hand)), int(max(yc_target, yc_hand))

    roi = depth_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
    try:
        max_depth = np.max(roi)
    except ValueError:
        max_depth = -1

    print(handBB[7])
    print(targetBB[7])
    print(max_depth)

    print(f"{xc_hand},{yc_hand},{xc_target},{yc_target}")
    print(depth_img.shape, roi.shape)
    if yc_hand < 480 and xc_hand < 640:
        print(depth_img[int(yc_hand), int(xc_hand)])

    """xyxy = xywh2xyxy(np.array(roi))
    label = "ROI"
    annotator.box_label(xyxy, label)

    # Display results
    im0 = annotator.result()
    cv2.imshow("ROI", im0) # original image only
    """

    baseline_depth_intensity = max(vibration_intensities.values())

    #if max_depth > handBB[7]:
    if max_depth < handBB[7]:
        print("object in line of movement")
        depth_intensity = round(-baseline_depth_intensity/5) * 5

    # Otherwise check if hand is closer or further than the target and set depth intensity accordingly
    else:
        depth_distance = handBB[7] - targetBB[7]
        if isinstance(depth_distance, (int, float, np.integer, np.floating)) and not(np.isnan(depth_distance)):
            if depth_distance > 0: #move forward
                depth_intensity = min(int(10000/depth_distance), baseline_depth_intensity) # d<=10 -> 100, d=1000 -> 10
            elif depth_distance < 0: #move backwards
                depth_intensity = max(int(10000/depth_distance), -baseline_depth_intensity) # d<=10 -> -100, d=1000 -> -10
            depth_intensity = round(depth_intensity/5) * 5 # steps in 5, so users can feel the change (can be replaced by a calibration value later for personalization)
        else:
            depth_intensity = 0 # placeholder
    
    return int(right_intensity), int(left_intensity), int(top_intensity), int(bottom_intensity), depth_intensity


def check_overlap(handBB, targetBB, frozen_x, frozen_y, freezed_width, freezed_height, freezed=False):

    # Get BB information
    hand_x, hand_y, hand_w, hand_h = handBB[:4]
    if freezed:
        target_x, target_y, target_w, target_h = frozen_x, frozen_y, freezed_width, freezed_height
    else:
        target_x, target_y = targetBB[:2]
        target_w, target_h = targetBB[2:4]

    # First iteration
    if freezed_width == -1 or freezed_height == -1:
        tbbw, tbbh = targetBB[2:4]
        return False, target_x, target_y, tbbw, tbbh, freezed

    # Calculate BB bounds to check for overlap
    hand_right = hand_x + hand_w//2
    hand_left = hand_x - hand_w//2
    hand_top = hand_y - hand_h//2
    hand_bottom = hand_y + hand_h//2

    target_right = target_x + target_w//2
    target_left = target_x - target_w//2
    target_top = target_y - target_h//2
    target_bottom = target_y + target_h//2

    # all cases of touching any side + handBB inside targetBB
    touched_left = hand_right >= target_left and hand_left <= target_left and hand_top <= target_bottom and hand_bottom >= target_top
    touched_right = hand_left <= target_right and hand_right >= target_right and hand_top <= target_bottom and hand_bottom >= target_top
    touched_top = hand_bottom >= target_top and hand_top <= target_top and hand_right >= target_left and hand_left <= target_right
    touched_bottom = hand_top <= target_bottom and hand_bottom >= target_bottom and hand_right >= target_left and hand_left <= target_right
    is_inside = hand_left >= target_left and hand_right <= target_right and hand_top >= target_top and hand_bottom <= target_bottom
    is_touched = touched_left or touched_right or touched_top or touched_bottom
    
    # If both BBs touch, keep freezed targetBB size
    if is_touched or is_inside:
        freezed = True
        #print('Touched!')
        # only if the center of the is in the targetBB send the grasp signal
        if (target_left <= hand_x <= target_right) and (target_top <= hand_y <= target_bottom):
            return True, frozen_x, frozen_y, freezed_width, freezed_height, freezed
        else:
            return False, frozen_x, frozen_y, freezed_width, freezed_height, freezed
    # Else, update targetBB size
    else:
        freezed = False
        #print('Updated BB size.')
        tbbw, tbbh = targetBB[2:4]
        return False, target_x, target_y, tbbw, tbbh, freezed


# GLOBALS
#max_intensity = calibrate_intensity()
vibration_intensities = {'bottom': 50,
                        'top': 50,
                        'left': 50,
                        'right': 50}
searching = False
prev_hand = None
prev_target = None
frozen_x = -1
frozen_y = -1
freezed_tbbw = -1
freezed_tbbh = -1
freezed = False
timer = 0

def navigate_hand(
        belt_controller, 
        bboxes,
        target_cls: str, 
        hand_clss: list,
        depth_img,
        vibration_intensities = {'bottom': 50,
                                'top': 50,
                                'left': 50,
                                'right': 50},
        mode = "grasping"):
    """ Function that navigates the hand to the target object. Handles cases when either hand or target is not detected.

    Args:
    - belt_controller -- belt controller object
    - bboxes -- object detections in current frame
    - target_cls -- the target object ID
    - hand_clss -- list of hand IDs
    - depth_img -- depth map of the currently processed frame
    - vibration_intensities -- intensitites of vibrations for each separate bracelet motor (range: 0 - 100)#
    - mode -- guiding algorithm selection, options: grasping, depth_navigation
    
    Returns:
    - overlapping -- information whether hand and target BBs are overlapping
    - frozen_target -- frozen target BB (updated up until occlusion of hand and target occurs)
    """

    #global max_intensity
    global searching
    global prev_hand
    global prev_target
    global frozen_x
    global frozen_y
    global freezed_tbbw
    global freezed_tbbh
    global freezed
    global timer
    overlapping = False

    # Search for object and hand with the highest prediction confidence
    ## Filter for hand detections
    bboxes_hands = [detection for detection in bboxes if detection[5] in hand_clss]
    hand = choose_detection(bboxes_hands, prev_hand)
    prev_hand = hand

    ## Filter for target detections
    bboxes_objects = [detection for detection in bboxes if detection[5] == target_cls]
    target = choose_detection(bboxes_objects, prev_target)
    prev_target = target
    print(target)
 
    if hand is not None and target is not None:
        # Get varying vibration intensities depending on angle from hand to target

        if mode == "grasping": # Standard grasping mode without use of depth signals
            right_int, left_int, top_int, bot_int, depth_int = get_intensity(hand, target, vibration_intensities, depth_img)

        elif mode == "depth_navigation": # Grasping utilizing depth information used in the A* algorithm for finding best trajectory
            astar_stop_condition, smoothing_distance = 50, 10
            obstacles_mask = map_obstacles(hand, target, depth_img)
            obstacles_between_hand_and_target = check_obstacles_between_points(hand, target, obstacles_mask, 1)
            #obstacles_between_hand_and_target = True

            if not obstacles_between_hand_and_target:
                print('No obstacles')
                right_int, left_int, top_int, bot_int, depth_int = get_intensity(hand, target, vibration_intensities, depth_img)
            
            else:
                print('Obstacles')
                path = astar(hand, target, obstacles_mask, 1, astar_stop_condition)
                #path = fast_pathfinding(hand, target, obstacles_mask, 1, astar_stop_condition)
                print(path)
                smoothed_path = smooth_path(path, smoothing_distance)
                right_int, left_int, top_int, bot_int, depth_int = get_intensity(smoothed_path[0], smoothed_path[1], vibration_intensities, depth_img)

        print(f'Vibration intensitites. Right: {right_int}, Left: {left_int}, Top: {top_int}, Bottom: {bot_int}.')

        # Check whether the hand is overlapping the target and freeze targetBB size if necessary (to avoid BB shrinking on occlusion)
        overlapping, frozen_x, frozen_y, freezed_tbbw, freezed_tbbh, freezed = check_overlap(hand, target, frozen_x, frozen_y, freezed_tbbw, freezed_tbbh, freezed)
        frozen_target = target.copy()
        frozen_target[0], frozen_target[1], frozen_target[2], frozen_target[3] = frozen_x, frozen_y, freezed_tbbw, freezed_tbbh

    # 1. Grasping
    if overlapping:
        searching = True
        if belt_controller:
            belt_controller.stop_vibration()
            belt_controller.send_pulse_command(
                            channel_index=1,
                            orientation_type=BeltOrientationType.BINARY_MASK,
                            orientation=0b111100,
                            intensity=abs(depth_int),
                            on_duration_ms=150,
                            pulse_period=500,
                            pulse_iterations=5,
                            series_period=5000,
                            series_iterations=1,
                            timer_option=BeltVibrationTimerOption.RESET_TIMER,
                            exclusive_channel=False,
                            clear_other_channels=False
                            )
        print("G R A S P !")
        return overlapping, frozen_target


    # 2. Guidance
    if hand is not None and target is not None:
        searching = True

        print("depth intensity: " + str(depth_int))
        if depth_int > 0:
            print("move forward")
        elif depth_int < 0:
            print("move backwards")

        if belt_controller:
            """
            # All motors vibrate with varying intensity
            belt_controller.vibrate_at_angle(120, intensity=right_int)
            belt_controller.vibrate_at_angle(45, intensity=left_int)
            belt_controller.vibrate_at_angle(90, intensity=top_int)
            belt_controller.vibrate_at_angle(60, intensity=bot_int)
            #belt_controller.vibrate_at_angle(0, intensity=depth_int)
            """
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=right_int,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=120,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=1,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=left_int,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=45,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=2,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=top_int,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=90,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=3,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=bot_int,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=60,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
        return overlapping, frozen_target


    # 3. Target is located and hand can be moved into the frame
    if target is not None:
        timer += 1
        if belt_controller and searching:
            searching = False
            belt_controller.stop_vibration()
            belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60, # bottom motor
                        intensity=max(vibration_intensities.values())//6,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=5,
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
        # reset searching flag to send command again
        if timer >= 50:
            searching = True
            timer = 0
        print('Target found.', searching, timer)
        return overlapping, target
    

    # 4. Target is not in the frame yet.
    else:
        timer = 0
        searching = True
        print('Target not found yet.')
        return overlapping, None