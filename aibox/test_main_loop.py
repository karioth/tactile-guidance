#!/usr/bin/env python3
"""
Test function to verify the main detection loop works as expected.
Checks that Detection tuples are properly formatted and returned.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add project paths for imports
file = Path(__file__).resolve()
root = file.parents[0]
sys.path.append(str(root))
sys.path.append(str(root) + '/vision_bridge')

from controller import TaskController
from gsam2_wrapper import GSAM2Wrapper


def test_detection_tuple_format(detection_tuple):
    """
    Test that a detection tuple has the correct format:
    (xc, yc, w, h, track_id, class_id, conf, depth)
    """
    if not isinstance(detection_tuple, (list, np.ndarray)):
        return False, f"Detection should be list/ndarray, got {type(detection_tuple)}"
    
    if len(detection_tuple) != 8:
        return False, f"Detection should have 8 elements, got {len(detection_tuple)}"
    
    # Check numeric types
    for i, val in enumerate(detection_tuple):
        if not isinstance(val, (int, float, np.number)):
            return False, f"Element {i} should be numeric, got {type(val)}"
    
    # Basic range checks
    xc, yc, w, h, track_id, class_id, conf, depth = detection_tuple
    if w < 0 or h < 0:
        return False, f"Width/height should be positive, got w={w}, h={h}"
    
    if not (0 <= conf <= 1):
        return False, f"Confidence should be in [0,1], got {conf}"
    
    return True, "Detection tuple format is correct"


def test_gsam2_wrapper():
    """Test GSAM2Wrapper directly."""
    print("Testing GSAM2Wrapper...")
    
    try:
        # Create wrapper instance
        wrapper = GSAM2Wrapper(device="cpu")  # Use CPU for testing
        
        # Create dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set prompt
        wrapper.set_prompt(dummy_frame, "coffee cup")
        
        # Track (may return empty list if no detections)
        detections = wrapper.track(dummy_frame)
        
        print(f"GSAM2Wrapper returned {len(detections)} detections")
        
        # Test detection format if any returned
        for i, det in enumerate(detections):
            is_valid, message = test_detection_tuple_format(det)
            if not is_valid:
                print(f"❌ Detection {i} format error: {message}")
                return False
            print(f"✅ Detection {i} format valid: {det}")
        
        print("✅ GSAM2Wrapper test passed")
        return True
        
    except Exception as e:
        print(f"❌ GSAM2Wrapper test failed: {e}")
        return False


def test_yolo_backend():
    """Test YOLO backend through TaskController."""
    print("Testing YOLO backend...")
    
    try:
        # Create TaskController with YOLO backend
        controller = TaskController(
            backend="yolo",
            weights_obj="yolov5s.pt",
            weights_hand="hand.pt", 
            source="vision_bridge/testingvid.mp4",
            device="cpu",  # Use CPU for testing
            view_img=False,
            nosave=True,
            classes_obj=[1, 39, 40, 41, 42, 45, 46, 47, 58, 74],
            classes_hand=[0, 1],
            conf_thres=0.7,
            iou_thres=0.45,
            imgsz=(640, 640),
            prompt="coffee cup"
        )
        
        print("✅ YOLO TaskController initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ YOLO backend test failed: {e}")
        return False


def test_gsam2_backend():
    """Test GSAM2 backend through TaskController."""
    print("Testing GSAM2 backend...")
    
    try:
        # Create TaskController with GSAM2 backend
        controller = TaskController(
            backend="gsam2",
            source="vision_bridge/testingvid.mp4",
            device="cpu",  # Use CPU for testing
            view_img=False,
            nosave=True,
            prompt="coffee cup"
        )
        
        print("✅ GSAM2 TaskController initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ GSAM2 backend test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("TESTING TACTILE GUIDANCE MAIN LOOP")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: GSAM2 Wrapper directly
    if test_gsam2_wrapper():
        tests_passed += 1
    
    # Test 2: YOLO backend
    if test_yolo_backend():
        tests_passed += 1
        
    # Test 3: GSAM2 backend
    if test_gsam2_backend():
        tests_passed += 1
    
    print("=" * 50)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 