import sys
import os

# Use the project file packages instead of the conda packages, i.e. add to system path for import
sys.path.append('/yolov5')
sys.path.append('/strongsort')
sys.path.append('/midas')
sys.path.append('/unidepth')

import argparse
import json
import controller
from bracelet import connect_belt, BraceletController


if __name__ == '__main__':

    # Parse arguments from CLI
    parser = argparse.ArgumentParser(description="Argument parser for bracelet tasks.")

    # Add arguments
    parser.add_argument(
        "-p", "--participant", 
        type=int, 
        required=True, 
        help="Participant number (randomize manually before)."
    )
    parser.add_argument(
        "-c", "--condition", 
        type=str, 
        required=True,
        choices=['grasping', 'multiple_objects', 'depth_navigation'],
        help="The task to be performed by the participant."
    )
    parser.add_argument(
        "-m", "--metric", 
        type=bool,
        default=True, 
        help="Whether metric or relative depth estimation should be used. Default: metric."
    )
    parser.add_argument(
        "--mock_navigate", 
        type=bool,
        default=False, 
        help="Whether to use mock navigation without a bracelet (for debugging)."
    )
    parser.add_argument(
        "--backend",
        choices=["yolo", "gsam2"],
        default="yolo",
        help="Vision backend (default: yolo).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="vision_bridge/testingvid.mp4",
        help="Image/video file path or camera index (default: testing video)",
    )
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Do not save annotated images/videos (default: save)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display live annotated window (default: hidden on HPC)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="coffee cup",
        help="Text prompt for Grounding-DINO when --backend gsam2 is used.",
    )
    parser.add_argument(
        "--handedness",
        choices=["left", "right"],
        default="right",
        help="Which hand is wearing the bracelet (default: right).",
    )
    
    
    # Parse the arguments
    args = parser.parse_args()
    participant = args.participant
    condition = args.condition
    metric = args.metric
    mock_navigate = args.mock_navigate
    source = args.source  # override default webcam with CLI path

    # Parameters
    weights_obj = 'yolov5s.pt'  # Object model weights path
    weights_hand = 'hand.pt' # Hands model weights path

    run_object_tracker = True if condition == 'multiple_objects' else False
    weights_tracker = 'osnet_x0_25_market1501.pt' # ReID weights path

    run_depth_estimator = True if condition == 'depth_navigation' else False
    weights_depth_estimator = 'v2-vits14' if metric else 'midas_v21_384' # v2-vits14, v1-cnvnxtl; midas_v21_384, dpt_levit_224

    belt_controller = None

    # Experiment controls
    target_objs = ['bottle', 'bicycle', 'potted plant', 'bowl', 'cup'] * 2 if condition == 'grasping' else ['bottle'] * 5
    output_path = 'results/' + f'{condition}/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        with open('results/calibration/' + f"calibration_participant_{participant}.json") as file:
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
                print('Please try to re-import the calibration file. Aborting.')
                sys.exit()

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
        task_controller = controller.TaskController(
            backend=args.backend,
            weights_obj=weights_obj,  # model_obj path or triton URL
            weights_hand=weights_hand,  # model_obj path or triton URL
            weights_tracker=weights_tracker,
            weights_depth_estimator=weights_depth_estimator,
            source=source,  # file/dir/URL/glob/screen/0(webcam)
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=args.view,  # --view to show window
            save_txt=False,  # save results to *.txtm)
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.7,  # confidence threshold
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=args.nosave,  # default saves unless --nosave
            classes_obj=[1,39,40,41,42,45,46,47,58,74],  # filter by class /  check coco.yaml file or coco_labels variable in this script
            classes_hand=[0,1], 
            #class_hand_nav=[80,81],
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=output_path+'video/',  # save results to project/name
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
            condition=condition,
            participant=participant,
            participant_vibration_intensities=participant_vibration_intensities,
            bracelet_controller=bracelet_controller,
            metric=metric,
            prompt=args.prompt,
            handedness=args.handedness)
        
        task_controller.run()
        
        # Print performance summary for GSAM2 backend
        if args.backend == "gsam2" and hasattr(task_controller, 'gsam2') and task_controller.gsam2 is not None:
            task_controller.gsam2.print_performance_summary()

    except KeyboardInterrupt:
        # Print performance summary even on interruption for GSAM2 backend
        if args.backend == "gsam2" and hasattr(task_controller, 'gsam2') and task_controller.gsam2 is not None:
            task_controller.gsam2.print_performance_summary()
        controller.close_app(belt_controller)

    # Print final performance summary for GSAM2 backend
    if args.backend == "gsam2" and hasattr(task_controller, 'gsam2') and task_controller.gsam2 is not None:
        task_controller.gsam2.print_performance_summary()
        
    # In the end, close all processes
    controller.close_app(belt_controller)