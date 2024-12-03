"""
This script is using code from the following sources:
- YOLOv5 🚀 by Ultralytics, AGPL-3.0 license, https://github.com/ultralytics/yolov5
- StrongSORT MOT, https://github.com/dyhBUPT/StrongSORT, https://pypi.org/project/strongsort/
- Youtube Tutorial "Simple YOLOv8 Object Detection & Tracking with StrongSORT & ByteTrack" by Nicolai Nielsen, https://www.youtube.com/watch?v=oDALtKbprHg
- https://github.com/zenjieli/Yolov5StrongSORT/blob/master/track.py, original: https://github.com/mikel-brostrom/yolo_tracking/commit/9fec03ddba453959f03ab59bffc36669ae2e932a
"""

# region Setup

# System
import sys
import os
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(str(parent_dir) + '/yolov5')
sys.path.append(str(parent_dir) + '/strongsort')
sys.path.append(str(parent_dir) + '/MiDaS')

os.chdir(parent_dir)

# Navigation
import controller

# Utility
import keyboard
from playsound import playsound
import threading

# endregion

# region Task

import numpy as np
import time

# Output data
import pandas as pd
import json

# Image processing
import cv2

# Object tracking
import torch
from labels import coco_labels # COCO labels dictionary
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, xywh2xyxy)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from strongsort.strong_sort import StrongSORT # there is also a pip install, but it has multiple errors
from ultralytics.utils.ops import non_max_suppression as nms

# DE
from midas.midas.model_loader import default_models, load_model
from midas.run import create_side_by_side, process

# Navigation
#from bracelet import navigate_hand, connect_belt
from bracelet import BraceletController, connect_belt

class GraspingTaskController(controller.TaskController):

    def save_output_data(self):

        df = pd.DataFrame(np.array(self.output_data).reshape(len(self.output_data)//3, 3))

        df.to_csv(self.output_path + f"grasping_task_participant_{self.participant}.csv")

    def print_output_data(self):

        df = pd.DataFrame(np.array(self.output_data).reshape(len(self.output_data)//3, 3))

        print(df)

    def experiment_trial_logic(self, trial_start_time, trial_end_time, pressed_key):

        # end trial
        if pressed_key in [ord('y'), ord('n')] and not self.ready_for_next_trial:
            trial_end_time = time.time()
            print(f'Trial time: {trial_end_time - trial_start_time}')
            self.output_data.append(trial_end_time - trial_start_time)
            self.output_data.append(chr(pressed_key))

            self.classes_obj = self.orig_classes_obj
            print(self.classes_obj)

            self.bracelet_controller.frozen = False
            
            if pressed_key == ord('y'):
                print("TRIAL SUCCESSFUL")
            elif pressed_key == ord('n'):
                print("TRIAL FAILED")
            
            if self.obj_index >= len(self.target_objs) - 1:
                print("ALL TARGETS COVERED")
                return "break"
            else:
                print("MOVING TO NEXT TARGET")
                self.obj_index += 1
                self.ready_for_next_trial = True
                self.class_target_obj = -1
        # start next trial
        elif pressed_key == ord('s') and self.ready_for_next_trial:
            print("STARTING NEXT TRIAL")
            self.target_entered = False
            self.ready_for_next_trial = False
            self.bracelet_controller.vibrate = True
        # end experiment
        elif pressed_key == ord('q'):
            if self.belt_controller:
                self.belt_controller.stop_vibration()
            return "break"
    
    def experiment_loop(self, save_dir, save_img, index_add, vid_path, vid_writer):

        print(f'\nSTARTING MAIN LOOP')

        # Initialize vars for tracking
        prev_frames = None
        curr_frames = None
        fpss = []
        outputs = []
        prev_outputs = np.array([])

        self.ready_for_next_trial = True
        self.target_entered = True # counter intuitive, but setting as True to wait for press of "s" button to start first trial
        self.class_target_obj = -1 # placeholder value not assigned to any specific object
        trial_start_time = -1 # placeholder initial value
        self.orig_classes_obj = self.classes_obj

        grasped = False

        # Data processing: Iterate over each frame of the live stream
        for frame, (path, im, im0s, vid_cap, _) in enumerate(self.dataset):

            # Start timer for FPS measure
            start = time.perf_counter()

            # Setup saving and visualization
            p, im0 = Path(path[0]), im0s[0].copy() # idx 0 is for first source (and we only have one source)
            save_path = str(save_dir / p.name)  # im.jpg
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names_obj))

            # Image pre-processing
            with self.dt[0]:
                image = torch.from_numpy(im).to(self.model_obj.device)
                #image = image.half() if self.model_obj.fp16 else image.float()  # uint8 to fp16/32
                image = image.half() if self.model_hand.fp16 else image.float()
                image /= 255  # 0 - 255 to 0.0 - 1.0
                if len(image.shape) == 3:
                    image = image[None]  # expand for batch dim

            # Object detection inference
            with self.dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred_target = self.model_obj(image, augment=self.augment, visualize=visualize) # YOLO11 runs nms by default
                pred_hand = self.model_hand(image, augment=self.augment, visualize=visualize)

            # Non-maximal supression
            with self.dt[2]:
                pred_target = non_max_suppression(pred_target, self.conf_thres, self.iou_thres, self.classes_obj, self.agnostic_nms, max_det=self.max_det) # list containing one tensor (n,6)
                #pred_target = nms(pred_target, self.conf_thres, self.iou_thres, self.classes_obj, self.agnostic_nms, max_det=self.max_det)
                pred_hand = non_max_suppression(pred_hand, self.conf_thres, self.iou_thres, self.classes_hand, self.agnostic_nms, max_det=self.max_det) # list containing one tensor (n,6)

            for hand in pred_hand[0]:
                if len(hand):
                    hand[5] += index_add # assign correct classID by adding len(coco_labels)

            # Camera motion compensation for tracker (ECC)
            if self.run_object_tracker:
                curr_frames = im0
                self.tracker.tracker.camera_update(prev_frames, curr_frames)
            
            # Initialize/clear detections
            xywhs = torch.empty(0,4)
            confs = torch.empty(0)
            clss = torch.empty(0)

            # Process object detections
            #print(pred_target[0])
            #print(pred_hand[0])
            #targets = torch.from_numpy(pred_target[0].orig_img)
            #print(targets)
            #preds = torch.cat((pred_target[0], pred_hand[0]), dim=0)
            preds = torch.cat((pred_target[0], pred_hand[0]), dim=0)
            if len(preds) > 0:
                preds[:, :4] = scale_boxes(im.shape[2:], preds[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(preds[:, :4])
                confs = preds[:, 4]
                clss = preds[:, 5]

            # Generate tracker outputs for navigation
            if self.run_object_tracker:
                
                # Update previous information
                outputs = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0) # takes xywhs, returns xyxys
                
                if not self.ready_for_next_trial:
                    hand_index_list = [hand + index_add for hand in self.classes_hand]
                    outputs = [output for output in outputs if output[5] in self.classes_obj + hand_index_list]

                # Add depth placeholder to outputs
                outputs = [np.append(bb, -1) for bb in outputs]
                """
                helper_list = []
                for bb in outputs:
                    bb = np.append(bb, -1)
                    helper_list.append(bb)
                outputs = helper_list
                """

                """
                # Get previous tracking information
                prev_track_ids = []
                if prev_outputs.size > 0:
                    prev_track_ids = prev_outputs[:, 4].tolist()
                
                # Get current tracking information
                tracks = self.tracker.tracker.tracks
                track_ids = []
                if len(outputs) > 0:
                    track_ids = np.array(outputs)[:, 4].tolist()

                # Revive previous information if necessary (get KF prediction for missing detections)
                tracker_ids = [track.track_id for track in tracks]
                # if there are more tracks than detections
                if 0 < len(prev_track_ids) > len(track_ids):
                    diff_ids = list(set(prev_track_ids) - set(track_ids))
                    for diff_id in diff_ids:
                        revivable_track = next((track for track in tracks if track.track_id == diff_id), None)
                        if revivable_track is not None and diff_id in tracker_ids and revivable_track.state == 2:
                            bbox_pred = revivable_track.mean[:4]
                            if 0 <= bbox_pred[2] <= 1:
                                bbox_pred[2] = bbox_pred[2] * bbox_pred[3] # xyah to xywh (or similar, but aspect to w)
                            bbox_pred = np.array(bbox_pred)
                            bbox_pred_xyxy = xywh2xyxy(bbox_pred).tolist() # convert xywh to xyxy, so all tracks in outputs can be converted together back to xywh
                            idx = np.where(prev_outputs[:, 4] == diff_id)[0]
                            if idx.size > 0:
                                prev_info = prev_outputs[idx[0], 4:].tolist()
                                # potentially set manual values for detection confidence and depth here
                                revived_detection = np.array(bbox_pred_xyxy + prev_info)
                                if isinstance(outputs, np.ndarray):
                                    outputs = outputs.tolist()
                                outputs.append(revived_detection)
                """

                # Convert BBs to xywh
                for bb in outputs:
                    bb[:4] = xyxy2xywh(bb[:4])

            # without tracking
            else:
                outputs = np.array(preds.cpu())
                outputs = np.insert(outputs, 4, -1, axis=1) # insert track_id placeholder
                outputs[:, [5, 6]] = outputs[:, [6, 5]] # switch cls and conf columns for alignment with tracker
                outputs = [np.append(bb, -1) for bb in outputs]

                for bb in outputs:
                    bb[:4] = xyxy2xywh(bb[:4])

            # Calculate difference between current and previous frame
            if prev_frames is not None:
                img_gr_1, img_gr_2 = cv2.cvtColor(curr_frames, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frames, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(img_gr_1, img_gr_2)
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)
                #print(f'Frames mean difference: {mean_diff}, SD: {std_diff}')
                if mean_diff > 30: # Big change between frames
                    #print('High change between frames. Resetting predictions.')
                    outputs = []
                #cv2.imshow('Diff',diff)
                #cv2.waitKey(0)

            # Depth estimation (automatically skips revived bbs)
            if not self.run_depth_estimator:
                depth_img = None

            else:
                depth_img, outputs = controller.get_depth(im0, self.transform, self.device, self.model, self.depth_estimator, self.net_w, self.net_h, vis=False, bbs=outputs)

            # Set current tracking information as previous info
            prev_outputs = np.array(outputs)

            # Get FPS
            end = time.perf_counter()
            runtime = end - start
            fps = 1 / runtime
            fpss.append(fps)
            prev_frames = curr_frames
            
        # endregion

        # region main navigation

            # Get the target object class
            if not self.target_entered:
                if self.manual_entry:
                    user_in = "n"
                    while user_in == "n":
                        print("These are the available objects:")
                        print(coco_labels)
                        target_obj_verb = input('Enter the object you want to target: ')

                        if target_obj_verb in coco_labels.values():
                            user_in = input("Selected object is " + target_obj_verb + ". Correct? [y,n]")
                            self.class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                            file = f'resources/sound/{target_obj_verb}.mp3'
                            #playsound(str(file))
                            # Start trial time measure (end in navigate_hand(...))
                        else:
                            print(f'The object {target_obj_verb} is not in the list of available targets. Please reselect.')
                else:
                    target_obj_verb = self.target_objs[self.obj_index]
                    self.class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                    file = f'resources/sound/{target_obj_verb}.mp3'
                    self.output_data.append(self.class_target_obj)
                    #playsound(str(file))

                self.target_entered = True
                self.classes_obj = [self.class_target_obj]
                print(self.classes_obj)
                grasped = False
                trial_start_time = time.time()
                vibration_timer = None

            # Navigate the hand based on information from last frame and current frame detections
            if not grasped:
                grasped, curr_target = self.bracelet_controller.navigate_hand(self.belt_controller, outputs, self.class_target_obj, [hand + index_add for hand in self.classes_hand], depth_img, self.participant_vibration_intensities)
            else: # if grasping signal was sent stop navigation process and reset target
                if vibration_timer is None:
                    vibration_timer = time.time()
                    grasped, curr_target = True, None
                elif vibration_timer > 0:
                    if time.time() - vibration_timer > 1.5:
                        if self.belt_controller:
                            self.belt_controller.stop_vibration()
                        vibration_timer = -1

        # region visualization
            # Write results
            for *xywh, obj_id, cls, conf, depth in outputs:
                id, cls = int(obj_id), int(cls)
                xyxy = xywh2xyxy(np.array(xywh))
                if save_img or self.save_crop or self.view_img:
                    label = None if self.hide_labels else (f'ID: {id} {self.master_label[cls]}' if self.hide_conf else (f'ID: {id} {self.master_label[cls]} {conf:.2f} {depth:.2f}'))
                    annotator.box_label(xyxy, label, color=colors(cls, True))

            # Target BB
            if curr_target is not None:
                #print(curr_target)
                for *xywh, obj_id, cls, conf, depth in [curr_target]:
                    xyxy = xywh2xyxy(np.array(xywh))
                    if save_img or self.save_crop or self.view_img:
                        label = None if self.hide_labels else 'Target object'
                        annotator.box_label(xyxy, label, color=(0,0,0))

            # Display results
            im0 = annotator.result()
            if self.view_img:
                cv2.putText(im0, f'FPS: {int(fps)}, Avg: {int(np.mean(fpss))}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 1)
                #side_by_side = create_side_by_side(im0, depth_img, False) # original image & depth side-by-side
                #cv2.imshow("AIBox & Depth", side_by_side)
                cv2.imshow("AIBox", im0)
                cv2.setWindowProperty("AIBox", cv2.WND_PROP_TOPMOST, 1)

                pressed_key = cv2.waitKey(1)

                trial_end_time = time.time()

                trial_info = self.experiment_trial_logic(trial_start_time, trial_end_time, pressed_key)
                
                if trial_info == "break":
                    try:
                        task_controller.print_output_data()
                        task_controller.save_output_data()
                    except ValueError:
                        pass
                    break

            # Save results
            if save_img:
                if self.dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[0] != save_path:  # new video
                        vid_path[0] = save_path
                        if isinstance(vid_writer[0], cv2.VideoWriter):
                            vid_writer[0].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 8, im0.shape[1], im0.shape[0] # int(np.mean(fpss))
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[0].write(im0)
            
        # endregion

# endregion

# region Main

if __name__ == '__main__':

    #check_requirements(requirements='../requirements.txt', exclude=('tensorboard', 'thop'))
    
    weights_obj = 'yolov5s.pt'  # Object model weights path
    weights_hand = 'hand.pt' # Hands model weights path
    weights_tracker = 'osnet_x0_25_market1501.pt' # ReID weights path
    depth_estimator = 'midas_v21_small_256' # depth estimator model type (weights are loaded automatically!), 
                                      # e.g.'midas_v21_small_256', ('dpt_levit_224', 'dpt_swin2_tiny_256',) 'dpt_large_384'
    source = '1' # image/video path or camera source (0 = webcam, 1 = external, ...)
    mock_navigate = True # Navigate without the bracelet using only print commands
    belt_controller = None
    run_object_tracker = False
    run_depth_estimator = False

    # EXPERIMENT CONTROLS

    target_objs = ['bottle', 'clock', 'potted plant'] * 3

    participant = 1
    output_path = str(parent_dir) + '/results/'

    try:
        with open(output_path + f"calibration_participant_{participant}.json") as file:
            participant_vibration_intensities = json.load(file)
        print('Calibration intensities loaded succesfully.')
    except:
        while True:
            continue_with_baseline = input('Error while loading the calibration file. Do you want to continue with baseline intensity of 50 for each vibromotor? (y/n)')
            if continue_with_baseline == 'y':
                participant_vibration_intensities = {'bottom': 50,
                                                     'top': 50,
                                                     'left': 50,
                                                     'right': 50,}
                break
            elif continue_with_baseline == 'n':
                sys.exit()

    #

    print(f'\nLOADING CAMERA AND BRACELET')

    # Check camera connection
    try:
        source = str(source)
        print('Camera connection successful')
    except:
        print('Cannot access selected source. Aborting.')
        sys.exit()

    # Check bracelet connection
    if not mock_navigate:
        connection_check, belt_controller = connect_belt()
        if connection_check:
            print('Bracelet connection successful.')
        else:
            print('Error connecting bracelet. Aborting.')
            sys.exit()

    try:
        bracelet_controller = BraceletController(vibration_intensities=participant_vibration_intensities)

        task_controller = GraspingTaskController(weights_obj=weights_obj,  # model_obj path or triton URL # ROOT
                        weights_hand=weights_hand,  # model_obj path or triton URL # ROOT
                        weights_tracker=weights_tracker, # ROOT
                        depth_estimator=depth_estimator,
                        source=source,  # file/dir/URL/glob/screen/0(webcam) # ROOT
                        iou_thres=0.45,  # NMS IOU threshold
                        max_det=1000,  # maximum detections per image
                        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                        view_img=True,  # show results
                        save_txt=False,  # save results to *.txtm)
                        imgsz=(640, 640),  # inference size (height, width)
                        conf_thres=0.7,  # confidence threshold
                        save_conf=False,  # save confidences in --save-txt labels
                        save_crop=False,  # save cropped prediction boxes
                        nosave=True,  # do not save images/videos
                        classes_obj=[1,39,40,41,42,45,46,47,58,74],  # filter by class /  check coco.yaml file or coco_labels variable in this script
                        classes_hand=[0,1], 
                        #class_hand_nav=[80,81],
                        agnostic_nms=False,  # class-agnostic NMS
                        augment=False,  # augmented inference
                        visualize=False,  # visualize features
                        update=False,  # update all models
                        project='runs/detect',  # save results to project/name # ROOT
                        name='video',  # save results to project/name
                        exist_ok=False,  # existing project/name ok, do not increment
                        line_thickness=3,  # bounding box thickness (pixels)
                        hide_labels=False,  # hide labels
                        hide_conf=False,  # hide confidences
                        half=False,  # use FP16 half-precision inference
                        dnn=False,  # use OpenCV DNN for ONNX inference
                        vid_stride=1,  # video frame-rate stride_obj
                        manual_entry=False, # True means you will control the exp manually versus the standard automatic running
                        run_object_tracker=run_object_tracker,
                        run_depth_estimator=run_depth_estimator,
                        mock_navigate=mock_navigate,
                        belt_controller=belt_controller,
                        tracker_max_age=60,
                        tracker_n_init=5,
                        target_objs=target_objs,
                        output_data=[],
                        output_path=output_path,
                        participant=participant,
                        participant_vibration_intensities=participant_vibration_intensities,
                        bracelet_controller=bracelet_controller) # debugging
        
        task_controller.run()

    except KeyboardInterrupt:
        controller.close_app(belt_controller)
    
    # In the end, kill everything
    controller.close_app(belt_controller)

# endregion