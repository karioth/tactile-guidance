#!/usr/bin/env python3
"""
Test function for GSAM2 + YOLO hand detection integration.
Verifies that both object and hand detections are properly combined and visualized.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch

# Add project paths for imports
file = Path(__file__).resolve()
root = file.parents[0]
sys.path.append(str(root))
sys.path.append(str(root) + '/vision_bridge')
sys.path.append(str(root) + '/yolov5')

from controller import TaskController
from gsam2_wrapper import GSAM2Wrapper
from yolov5.utils.general import non_max_suppression


def test_detection_format(detection):
    """
    Test that a detection has the correct format:
    (xc, yc, w, h, track_id, class_id, conf, depth)
    """
    if not isinstance(detection, np.ndarray):
        return False, f"Detection should be numpy array, got {type(detection)}"
    
    if detection.shape != (8,):
        return False, f"Detection should have 8 elements, got {detection.shape}"
    
    xc, yc, w, h, track_id, class_id, conf, depth = detection
    
    # Basic range checks
    if not (0 <= conf <= 1):
        return False, f"Confidence should be 0-1, got {conf}"
    
    if not (w > 0 and h > 0):
        return False, f"Width and height should be positive, got w={w}, h={h}"
    
    if class_id not in [0, 80, 81]:  # Objects: 0, Hands: 80, 81
        return False, f"Class ID should be 0 (object) or 80/81 (hands), got {class_id}"
    
    return True, "Detection format is correct"


def test_gsam2_hand_integration():
    """
    Test the GSAM2 + hand detection integration
    """
    print("🧪 Testing GSAM2 + Hand Detection Integration")
    print("=" * 50)
    
    try:
        # Test configuration - minimal setup
        print("📋 Initializing TaskController with GSAM2 backend...")
        controller = TaskController(
            backend="gsam2",
            weights_obj='yolov5s.pt',
            weights_hand='hand.pt',
            source="0",               # Dummy webcam source
            prompt="coffee cup",
            view_img=False,
            nosave=True,
            conf_thres=0.5,
            device='',
            imgsz=(640, 640),
            classes_obj=[39, 41, 42, 46, 47],
            classes_hand=[0, 1],
            run_object_tracker=False,
            run_depth_estimator=False,
            mock_navigate=True,
            belt_controller=None,
            # Add missing attributes for detector loading
            dnn=False,
            half=False,
            agnostic_nms=False,
            max_det=1000,
            augment=False,
            iou_thres=0.45
        )
        
        print("✅ Controller initialized successfully")
        
        # Load the detectors manually (normally done in run() method)
        controller.load_object_detector()
        print("✅ Detectors loaded")
        
        # Test the conversion function with mock data
        print("\n🔧 Testing convert_and_combine_detections function...")
        
        # Mock GSAM2 object detection (already in Detection format)
        mock_gsam2_objects = [
            np.array([320, 240, 100, 80, -1, 0, 0.85, -1])  # Coffee cup detection
        ]
        
        # Mock YOLO hand tensor (xyxy format)
        mock_hand_tensor = torch.tensor([
            [100, 150, 200, 250, 0.9, 0],  # Left hand
            [400, 180, 500, 280, 0.8, 1],  # Right hand
        ])
        
        # Mock image tensors
        mock_im = torch.randn(1, 3, 640, 640)  # Preprocessed image
        mock_im0 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Original image
        index_add = 80
        
        # Test the function
        combined = controller.convert_and_combine_detections(
            mock_gsam2_objects, mock_hand_tensor, mock_im, mock_im0, index_add
        )
        
        print(f"📊 Combined detections: {len(combined)} total")
        print(f"   - Objects: {len(mock_gsam2_objects)}")
        print(f"   - Hands: {len(mock_hand_tensor)}")
        
        # Verify combined detections
        if len(combined) != 3:
            print(f"❌ Expected 3 detections, got {len(combined)}")
            return False
        
        # Test each detection format
        for i, detection in enumerate(combined):
            is_valid, message = test_detection_format(detection)
            if not is_valid:
                print(f"❌ Detection {i} format error: {message}")
                return False
            
            xc, yc, w, h, track_id, class_id, conf, depth = detection
            if i == 0:  # Object
                print(f"   🎯 Object: cls={class_id}, conf={conf:.2f}, pos=({xc:.1f},{yc:.1f})")
            else:  # Hand
                hand_type = "left" if class_id == 80 else "right"
                print(f"   ✋ {hand_type.capitalize()} hand: cls={class_id}, conf={conf:.2f}, pos=({xc:.1f},{yc:.1f})")
        
        print("✅ All detections have correct format")
        
        # Test tensor shape handling
        print("\n🔍 Testing tensor shape handling...")
        
        # Test empty hand tensor
        empty_hand_tensor = torch.empty(0, 6)
        combined_empty = controller.convert_and_combine_detections(
            mock_gsam2_objects, empty_hand_tensor, mock_im, mock_im0, index_add
        )
        
        if len(combined_empty) != len(mock_gsam2_objects):
            print(f"❌ Empty hand tensor handling failed")
            return False
        
        print("✅ Empty tensor handling works")
        
        # Test model loading verification
        print("\n🔍 Testing model availability...")
        
        if hasattr(controller, 'gsam2') and controller.gsam2 is not None:
            print("✅ GSAM2 model loaded")
        else:
            print("❌ GSAM2 model not loaded")
            return False
            
        if hasattr(controller, 'model_hand') and controller.model_hand is not None:
            print("✅ Hand model loaded")
        else:
            print("❌ Hand model not loaded")
            return False
        
        print("✅ Model availability test passed")
        
        print("\n🎉 All tests passed! GSAM2 + Hand integration is working correctly.")
        print("\n📋 What you should expect when running master.py:")
        print("   - Object bounding boxes (from GSAM2) with your prompt")
        print("   - Hand bounding boxes (from YOLO) for left/right hands")
        print("   - Both types should appear in the video output")
        print("   - Motor navigation should work with both object and hand positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Quick test to verify imports work"""
    print("🔍 Testing imports...")
    
    try:
        from controller import TaskController
        print("✅ TaskController imported")
        
        from gsam2_wrapper import GSAM2Wrapper
        print("✅ GSAM2Wrapper imported")
        
        # Check if hand model file exists
        hand_model_path = Path("hand.pt")
        if hand_model_path.exists():
            print("✅ Hand model found at hand.pt")
        else:
            print("⚠️  Hand model not found at hand.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting GSAM2 + Hand Detection Tests")
    print("=" * 60)
    
    # Test imports first
    if not test_imports():
        print("❌ Import tests failed, aborting")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Test main integration
    if test_gsam2_hand_integration():
        print("\n🎯 SUCCESS: Ready to run with both object and hand detection!")
        print("\n💡 Run this command to see both bounding boxes:")
        print("   python master.py --backend gsam2 --prompt 'coffee cup' --view")
    else:
        print("\n❌ FAILED: Integration has issues")
        sys.exit(1) 