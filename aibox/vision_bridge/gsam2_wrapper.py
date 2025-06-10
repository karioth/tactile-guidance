from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum
from collections import deque
import time
import cv2

# **SAM-2 Dtype Optimizations (from official demos)**
# Enable bfloat16 autocast for better performance on modern GPUs
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Enable TF32 on Ampere GPUs (RTX 30xx/40xx series) for additional speedup
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("🚀 Enabled TF32 optimizations for Ampere GPU")

# Add Grounded-SAM-2 to path for sam2 imports
_GROUNDED_SAM2_PATH = Path(__file__).resolve().parent.parent / "Grounded-SAM-2"
if str(_GROUNDED_SAM2_PATH) not in sys.path:
    sys.path.insert(0, str(_GROUNDED_SAM2_PATH))

# Grounding DINO helper (tiny Swin-T backbone)
from groundingdino.util.inference import Model as GDINOModel
from torchvision.ops import box_convert

# SAM-2 imports
from sam2.build_sam import build_sam2_video_predictor

# **DEBUG: Import SAM-2 attention settings**
from sam2.utils.misc import get_sdpa_settings

# Supervision for mask-to-xyxy conversion
import supervision as sv


class TrackingState(Enum):
    """Simplified state machine for SAM-2 tracking workflow."""
    WAITING_FOR_HAND = "waiting_for_hand"       # Looking for hand
    HAND_READY = "hand_ready"                   # Hand found, ready for object prompt  
    SEARCHING_OBJECT = "searching_object"       # Hand tracked, searching for object
    TRACKING_BOTH = "tracking_both"             # Both hand and object tracked


class SimpleLossTracker:
    """Simple tracking loss detection - only what we actually use."""
    
    def __init__(self, max_loss_frames: int = 20):
        self.max_loss_frames = max_loss_frames  # Consider lost after 20 consecutive missed frames (~0.67s at 30fps) - increased from 8 for SAM-2's better temporal tracking
        self.object_loss_frames = 0
        self.hand_loss_frames = 0
        
    def update(self, has_object: bool, has_hand: bool) -> None:
        """Update loss counters."""
        # Update object tracking
        if has_object:
            self.object_loss_frames = 0  # Reset loss counter
        else:
            self.object_loss_frames += 1
            
        # Update hand tracking
        if has_hand:
            self.hand_loss_frames = 0  # Reset loss counter
        else:
            self.hand_loss_frames += 1
    
    def is_object_lost(self) -> bool:
        """Simple check if object tracking is lost."""
        return self.object_loss_frames >= self.max_loss_frames
    
    def is_hand_lost(self) -> bool:
        """Simple check if hand tracking is lost."""
        return self.hand_loss_frames >= self.max_loss_frames
    
    def get_loss_status(self) -> dict:
        """Get current loss status for both targets (compatibility method)."""
        return {
            "object_lost": self.is_object_lost(),
            "hand_lost": self.is_hand_lost(),
            "object_loss_frames": self.object_loss_frames,
            "hand_loss_frames": self.hand_loss_frames,
        }
    
    def reset(self) -> None:
        """Reset all tracking counters."""
        self.object_loss_frames = 0
        self.hand_loss_frames = 0


class TrackingStateMachine:
    """Hand-first state machine for SAM-2 tracking workflow."""
    
    def __init__(self, loss_tracker: SimpleLossTracker):
        self.loss_tracker = loss_tracker
        self.state = TrackingState.WAITING_FOR_HAND
        self.state_entry_time = time.time()
        self.object_search_attempts = 0
        self.max_object_search_attempts = 10  # ~5s at 30fps
        
    def update_state(self, has_object: bool, has_hand: bool) -> TrackingState:
        """Hand-first state machine with essential transitions."""
        object_lost = self.loss_tracker.is_object_lost()
        hand_lost = self.loss_tracker.is_hand_lost()
        
        if self.state == TrackingState.WAITING_FOR_HAND:
            if has_hand:
                self._transition_to(TrackingState.HAND_READY)
                
        elif self.state == TrackingState.HAND_READY:
            if hand_lost:
                self._transition_to(TrackingState.WAITING_FOR_HAND)
                
        elif self.state == TrackingState.SEARCHING_OBJECT:
            if hand_lost:
                self._transition_to(TrackingState.WAITING_FOR_HAND)
                self.object_search_attempts = 0
            elif has_object and has_hand:
                self._transition_to(TrackingState.TRACKING_BOTH)
                self.object_search_attempts = 0
                    
        elif self.state == TrackingState.TRACKING_BOTH:
            if hand_lost:
                self._transition_to(TrackingState.WAITING_FOR_HAND)
            elif object_lost:
                self._transition_to(TrackingState.HAND_READY)
        
        return self.state
    
    def start_object_search(self) -> bool:
        """Start searching for object."""
        if self.state == TrackingState.HAND_READY:
            self._transition_to(TrackingState.SEARCHING_OBJECT)
            self.object_search_attempts = 0
            return True
        return False
    
    def is_ready_for_object_prompt(self) -> bool:
        """Check if ready for object prompt."""
        return self.state == TrackingState.HAND_READY
    
    def should_search_for_object(self) -> bool:
        """Check if actively searching for object."""
        return self.state == TrackingState.SEARCHING_OBJECT
    
    def is_fully_operational(self) -> bool:
        """Check if tracking both targets."""
        return self.state == TrackingState.TRACKING_BOTH
    
    def increment_search_attempt(self) -> bool:
        """Increment object search attempts and check if timeout reached.
        
        Returns:
            bool: True if search should continue, False if timeout reached
        """
        if self.state == TrackingState.SEARCHING_OBJECT:
            self.object_search_attempts += 1
            if self.object_search_attempts >= self.max_object_search_attempts:
                print("⏰ Object search timeout")
                self._transition_to(TrackingState.HAND_READY)
                self.object_search_attempts = 0
                return False
        return True
    
    def _transition_to(self, new_state: TrackingState) -> None:
        """Handle state transitions."""
        if new_state != self.state:
            self.state = new_state
            self.state_entry_time = time.time()
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.state = TrackingState.WAITING_FOR_HAND
        self.state_entry_time = time.time()


class GSAM2Wrapper:
    """Hand-first wrapper that exposes a YOLO-compatible interface
    around Grounding-DINO + SAM-2 tracking.

    Uses a hand-first workflow: detects and tracks the user's hand first,
    then accepts object prompts to search for and track objects.
    Uses Grounding-DINO for initial detection and SAM-2 for 
    temporal tracking with mask propagation.

    Public API
    ----------
    set_prompt(frame_rgb, text)
        Set object prompt for hand-first workflow.
    track(frame_rgb, depth_img=None) -> list[tuple]
        Return detection tuples in the format required by `bracelet.py`::

            (xc, yc, w, h, track_id, class_id, conf, depth)
    """

    # ---------------------------------------------------------------------
    # Construction helpers - Updated paths for new structure
    # ---------------------------------------------------------------------
    _CONF_PATH = (
        Path(__file__).resolve().parent.parent
        / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    _CKPT_PATH = (
        Path(__file__).resolve().parent.parent
        / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    )
    
    # SAM-2 paths (using tiny model for efficiency)
    _SAM2_CONFIG = "sam2/configs/sam2/sam2_hiera_t.yaml"
    _SAM2_CHECKPOINT = (
        Path(__file__).resolve().parent.parent
        / "Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"
    )

    def __init__(
        self,
        device: str | torch.device | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        default_prompt: str = "coffee cup",
        handedness: str = "right",
        frame_cache_limit: int = 100,  # Reset every 100 frames (~3.3s at 30fps) - reduced from 300 for better performance
    ) -> None:
        """Initialize GSAM2Wrapper with hand-first workflow.

        Parameters
        ----------
        device : str | torch.device | None
            Device for PyTorch model inference. If None, auto-detected.
        box_threshold : float
            Box detection threshold for Grounding-DINO (default: 0.35).
        text_threshold : float
            Text threshold for Grounding-DINO (default: 0.25).
        default_prompt : str
            Default object prompt for detection.
        handedness : str
            User handedness - "left" or "right" (default: "right").
        frame_cache_limit : int
            Maximum frames cached by SAM-2 before memory reset (default: 100).
        """
        # Device detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.handedness = handedness
        self.frame_cache_limit = frame_cache_limit

        # Initialize Grounding-DINO 
        self._gdino = GDINOModel(
            model_config_path=str(self._CONF_PATH),
            model_checkpoint_path=str(self._CKPT_PATH),
            device=str(self.device),
        )

        # Load SAM-2 video predictor for tracking
        # Change to Grounded-SAM-2 directory for config resolution
        grounded_sam2_dir = Path(__file__).resolve().parent.parent / "Grounded-SAM-2"
        original_cwd = os.getcwd()
        os.chdir(str(grounded_sam2_dir))
        
        try:
            self._sam2_video = build_sam2_video_predictor(
                "configs/sam2.1/sam2.1_hiera_t.yaml", 
                str(self._SAM2_CHECKPOINT),
                device=str(self.device)
            )
            print(f"✅ SAM-2 video predictor loaded successfully")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        # Initialize inference state for streaming mode
        self._inference_state = self._sam2_video.init_state(video_path=None)
        
        # Initialize empty image tensor for real-time streaming
        self._inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        self._inference_state["device"] = self.device  # Ensure device is set for streaming method

        self._prompt: str = default_prompt
        self._frame_count: int = 0
        self._object_prompt_provided: bool = False  # Track if user has provided object prompt

        # Initialize loss tracker
        self.loss_tracker = SimpleLossTracker()
        self.tracking_state_machine = TrackingStateMachine(self.loss_tracker)

        # Initialize tracking state
        self._sam2_primed = False
        self._tracked_object_id = None
        self._tracked_hand_id = None
        
        # Hand-first workflow state
        self._hand_initialized = False
        
        # **SIMPLIFIED: Single search counter - only used when actively searching**
        self._search_frame_counter = 0
        self._search_interval = 15  # Wait 20 frames between GDINO calls
        
        # Track last known positions for memory management
        self._last_hand_box = None
        self._last_object_box = None

        # Simple performance tracking
        self._total_frames_processed = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_prompt(self, frame_rgb: np.ndarray | None, text: str) -> None:
        """Set object prompt for hand-first workflow.
        
        This is equivalent to set_object_prompt() for hand-first workflow.

        Parameters
        ----------
        frame_rgb : np.ndarray | None
            RGB frame for object detection and SAM-2 priming. If None, only updates
            the text prompt without priming.
        text : str
            The free-form text prompt (e.g. "red bottle").
        """
        
        if frame_rgb is None:
            # Just update the prompt
            self._prompt = text
            print(f"🔄 Object prompt updated: '{text}' (no frame provided)")
            return
        
        # If hand not initialized, start hand tracking first
        if not self._hand_initialized:
            print("🤚 Starting hand tracking first...")
            self.start_hand_tracking(frame_rgb)
            # Set the object prompt for later
            self._prompt = text
            self._object_prompt_provided = True
            return
        
        # If ready for object prompt, start object search
        if self.is_ready_for_object_prompt():
            self.set_object_prompt(text)
            return
        
        # Otherwise just update the prompt
        self._prompt = text
        print(f"🔄 Object prompt updated: '{text}'")
    
    def _detect_and_prime_sam2(self, frame_rgb: np.ndarray, prompt: str, 
                              target_types: List[str] = ["object", "hand"],
                              reset_state: bool = False, 
                              preserve_existing: Optional[List[str]] = None) -> bool:
        """Unified detection and SAM-2 priming logic.
        
        Args:
            frame_rgb: RGB frame for detection
            prompt: Text prompt for Grounding-DINO
            target_types: List of target types to detect ["object", "hand"]
            reset_state: Whether to reset SAM-2 state before adding
            preserve_existing: List of existing targets to preserve during reset ["hand", "object"]
            
        Returns:
            bool: True if detection and priming successful
        """
        try:
            # Convert RGB to BGR for Grounding-DINO
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Call Grounding-DINO
            detections, labels = self._call_gdino_with_tracking(
                frame_bgr=frame_bgr,
                caption=prompt,
            )
            
            if len(detections.xyxy) == 0:
                return False
            
            # Parse detections based on target types
            object_candidates = []
            hand_candidates = []
            
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i])
                label = labels[i].lower() if i < len(labels) else ""
                
                detection_data = (x1, y1, x2, y2, conf, label)
                
                if "hand" in label and "hand" in target_types:
                    hand_candidates.append(detection_data)
                elif "object" in target_types and "hand" not in label:
                    object_candidates.append(detection_data)
            
            # Select best candidates
            best_object = None
            best_hand = None
            
            if "object" in target_types:
                best_object = self._select_best_candidate(object_candidates)
            if "hand" in target_types:
                best_hand = self._select_best_candidate(hand_candidates)
            
            # Check if we found required targets
            if "object" in target_types and best_object is None and "hand" in target_types and best_hand is None:
                return False
            if "object" in target_types and len(target_types) == 1 and best_object is None:
                return False
            if "hand" in target_types and len(target_types) == 1 and best_hand is None:
                return False
            
            # Handle SAM-2 state management
            if reset_state:
                # Reset state and preserve existing targets
                preserved_boxes = {}
                if preserve_existing:
                    if "hand" in preserve_existing and hasattr(self, '_last_hand_box') and self._last_hand_box is not None:
                        preserved_boxes["hand"] = self._last_hand_box.copy()
                    if "object" in preserve_existing and hasattr(self, '_last_object_box') and self._last_object_box is not None:
                        preserved_boxes["object"] = self._last_object_box.copy()
                
                # Reset SAM-2 state
                self._sam2_video.reset_state(self._inference_state)
                frame_idx = self._sam2_video.add_new_frame(self._inference_state, frame_rgb)
                
                # Re-add preserved targets first
                for target_type, box in preserved_boxes.items():
                    if target_type == "hand":
                        self._tracked_hand_id = 2
                        self._sam2_video.add_new_points_or_box(
                            inference_state=self._inference_state,
                            frame_idx=frame_idx,
                            obj_id=self._tracked_hand_id,
                            box=box,
                        )
                    elif target_type == "object":
                        self._tracked_object_id = 1
                        self._sam2_video.add_new_points_or_box(
                            inference_state=self._inference_state,
                            frame_idx=frame_idx,
                            obj_id=self._tracked_object_id,
                            box=box,
                        )
            else:
                # Normal state management
                self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
                frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
            
            # Add newly detected targets
            success_count = 0
            
            if best_object is not None:
                x1, y1, x2, y2, conf, label = best_object
                object_box = np.array([x1, y1, x2, y2])
                
                self._tracked_object_id = 1  # Object gets ID 1
                
                self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_object_id,
                    box=object_box,
                )
                
                self._last_object_box = object_box.copy()
                success_count += 1
            
            if best_hand is not None:
                x1, y1, x2, y2, conf, label = best_hand
                hand_box = np.array([x1, y1, x2, y2])
                
                self._tracked_hand_id = 2  # Hand gets ID 2
                
                self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_hand_id,
                    box=hand_box,
                )
                
                self._last_hand_box = hand_box.copy()
                success_count += 1
            
            # Update state
            self._sam2_primed = True
            self._frame_count = frame_idx + 1
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Unified detection failed: {e}")
            return False

    def _select_best_candidate(self, candidates: List[tuple]) -> Optional[tuple]:
        """Simplified candidate selection - just pick highest confidence."""
        if not candidates:
            return None
        
        # Simple confidence-based selection
        best_candidate = max(candidates, key=lambda x: x[4])  # x[4] is confidence
        
        return best_candidate

    def _call_gdino_with_tracking(self, frame_bgr: np.ndarray, caption: str) -> tuple:
        """Call GDINO for object detection.
        
        Args:
            frame_bgr: BGR frame for detection
            caption: Text prompt for detection
            
        Returns:
            tuple: (detections, labels) from GDINO
        """
        detections, labels = self._gdino.predict_with_caption(
            image=frame_bgr,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        
        return detections, labels

    def _should_attempt_detection(self, detection_type: str = "detection") -> bool:
        """Unified search timing logic to avoid duplication.
        
        Args:
            detection_type: Type of detection for logging ("hand", "object", "detection")
            
        Returns:
            bool: True if detection should be attempted this frame
        """
        # Attempt detection immediately on first frame (counter=0), then every 15 frames
        if self._search_frame_counter == 0 or self._search_frame_counter >= self._search_interval:
            print(f"🔍 Attempting {detection_type} detection...")
            self._search_frame_counter = 1  # Set to 1 after detection attempt
            return True
        else:
            # Count frames between detection attempts
            self._search_frame_counter += 1
            return False

    @torch.inference_mode()
    def track(
        self,
        frame_bgr: np.ndarray,
        depth_img: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Hand-first tracking using SAM-2 temporal tracking.

        Returns
        -------
        list[np.ndarray]
            List of detection tuples: [xc, yc, w, h, track_id, class_id, conf, depth]
            class_id: 0 = target object, 1 = hand
        """
        # Count every frame processed (including skipped ones)
        self._total_frames_processed += 1
        
        current_state = self.tracking_state_machine.state
        
        # **SIMPLIFIED: Handle based on current state, not workflow mode**
        if current_state == TrackingState.WAITING_FOR_HAND:
            results = self._handle_hand_initialization(frame_bgr, depth_img)
        
        elif current_state == TrackingState.HAND_READY:
            # **Auto-start object search if prompt was provided**
            if hasattr(self, '_prompt') and self._prompt and self._prompt.strip():
                print(f"🔍 Auto-starting object search for: '{self._prompt}'")
                if self.set_object_prompt(self._prompt):
                    # Successfully started object search, continue with SAM-2 tracking
                    pass
                else:
                    print("⚠️  Failed to auto-start object search")
                    
            results = self._track_with_sam2(frame_bgr, depth_img)
        
        elif current_state == TrackingState.SEARCHING_OBJECT:
            results = self._track_with_sam2(frame_bgr, depth_img)
            
            # **FIX: Check state after SAM-2 tracking to see if we transitioned to TRACKING_BOTH**
            current_state_after_sam2 = self.tracking_state_machine.state
            
            # Only attempt object detection if still in SEARCHING_OBJECT state
            if current_state_after_sam2 == TrackingState.SEARCHING_OBJECT:
                # Attempt object detection immediately on first frame (counter=0), then every 15 frames
                if self._should_attempt_detection("object"):
                    # Increment search attempt counter and check for timeout
                    if self.tracking_state_machine.increment_search_attempt():
                        detection_success = self._attempt_object_detection(frame_bgr)
                        # Counter is already handled by _should_attempt_detection()
                    else:
                        print("🔍 Search timeout reached, stopping attempts")
            else:
                print(f"✅ State transitioned to {current_state_after_sam2.value}, stopping object search")
        
        elif current_state == TrackingState.TRACKING_BOTH:
            results = self._track_with_sam2(frame_bgr, depth_img)
        
        else:
            # Fallback
            results = self._track_with_sam2(frame_bgr, depth_img) if self._sam2_primed else []
        
        return results
    
    def _handle_hand_initialization(self, frame_bgr: np.ndarray, depth_img: Optional[np.ndarray]) -> List[np.ndarray]:
        """Handle hand initialization state with frame-based detection intervals."""
        
        if not self._sam2_primed:
            # Attempt hand detection immediately on first frame (counter=0), then every 15 frames  
            if self._should_attempt_detection("hand"):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if self.start_hand_tracking(frame_rgb):
                    # Counter is already handled by _should_attempt_detection()
                    return self._track_with_sam2(frame_bgr, depth_img)
                else:
                    # Counter is already handled by _should_attempt_detection()
                    return []
            else:
                # Counter is already handled by _should_attempt_detection()
                return []
        else:
            # SAM-2 is primed, continue tracking
            return self._track_with_sam2(frame_bgr, depth_img)
    
    def _track_with_sam2(self, frame_bgr: np.ndarray, depth_img: Optional[np.ndarray]) -> List[np.ndarray]:
        """Track using SAM-2 temporal tracking."""
        try:
            # Convert BGR to RGB for SAM-2
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Check for memory reset before processing
            reset_occurred = self._check_memory_reset(frame_rgb)
            if reset_occurred and not self._sam2_primed:
                # Reset occurred and objects couldn't be preserved - return empty results
                # Let the state machine transition to recovery/re-initialization
                print("⚠️  Memory reset failed to preserve tracking - returning empty results")
                return []
            
            # Add new frame to SAM-2 inference state
            frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
            
            # Run SAM-2 inference on the frame
            frame_idx, tracked_obj_ids, video_res_masks = self._sam2_video.infer_single_frame(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
            )
            
            results = []
            
            # Process tracked masks and convert to Detection tuples
            for i, obj_id in enumerate(tracked_obj_ids):
                # Get mask for this object
                mask = video_res_masks[i] > 0.0  # Convert to boolean mask
                mask_np = mask[0].cpu().numpy()  # Shape: (H, W)
                
                # Convert mask to bounding box using supervision
                try:
                    # sv.mask_to_xyxy expects masks with shape (n, H, W)
                    masks_batch = mask_np[None, ...]  # Add batch dimension: (1, H, W)
                    xyxy_boxes = sv.mask_to_xyxy(masks_batch)  # Returns (n, 4) in xyxy format
                    
                    if len(xyxy_boxes) > 0:
                        x1, y1, x2, y2 = xyxy_boxes[0]  # Get first (and only) box
                        
                        # Convert xyxy to center format
                        xc = (x1 + x2) / 2
                        yc = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Determine class_id based on tracked object ID
                        if obj_id == self._tracked_object_id:
                            class_id = 0  # Target object
                            track_id = obj_id
                            # Preserve last known position for memory management
                            self._last_object_box = np.array([x1, y1, x2, y2])
                        elif obj_id == self._tracked_hand_id:
                            class_id = 1  # Hand
                            track_id = obj_id
                            # Preserve last known position for memory management
                            self._last_hand_box = np.array([x1, y1, x2, y2])
                        else:
                            # Unknown object ID, skip
                            continue
                        
                        # Use a high confidence for successfully tracked objects
                        conf = 0.8
                        
                        # Handle depth
                        depth_val = -1.0
                        if depth_img is not None:
                            # Simple depth sampling at object center
                            center_y, center_x = int(yc), int(xc)
                            if (0 <= center_y < depth_img.shape[0] and 
                                0 <= center_x < depth_img.shape[1]):
                                depth_val = float(depth_img[center_y, center_x])
                        
                        # Create Detection tuple
                        detection = np.array([xc, yc, w, h, track_id, class_id, conf, depth_val], dtype=float)
                        results.append(detection)
                        
                except Exception as e:
                    print(f"⚠️  Failed to convert mask to bbox for object {obj_id}: {e}")
                    continue
            
            self._frame_count += 1
            
            # Update state machine
            best_object = None
            best_hand = None
            
            for detection in results:
                if detection[5] == 0:  # class_id 0 = object
                    best_object = detection
                elif detection[5] == 1:  # class_id 1 = hand
                    best_hand = detection
            
            self.loss_tracker.update(best_object is not None, best_hand is not None)
            previous_state = self.tracking_state_machine.state
            self.tracking_state_machine.update_state(best_object is not None, best_hand is not None)
            
            # **NEW: Reset frame counters on state transitions**
            if previous_state != self.tracking_state_machine.state:
                if self.tracking_state_machine.state == TrackingState.WAITING_FOR_HAND:
                    self._search_frame_counter = 0
                    print("🔄 Reset search counter due to state transition")
            
            return results
            
        except Exception as e:
            print(f"⚠️  SAM-2 tracking failed: {e}")
            # On tracking failure, let state machine handle transitions based on empty results
            # This is simpler and more reliable than complex recovery attempts
            print("🔄 Returning empty results - state machine will handle transitions")
            return []
    
    def _check_memory_reset(self, frame_rgb: np.ndarray) -> bool:
        """Simple memory management - use original approach until SAM-2 API is clarified."""
        if self._inference_state["images"].shape[0] >= self.frame_cache_limit:
            return self._fallback_full_reset(frame_rgb)
        return False
    
    def _fallback_full_reset(self, frame_rgb: np.ndarray) -> bool:
        """Fallback full reset with tracking preservation (previous approach)."""
        # Preserve current tracking state if available
        current_hand_box = None
        current_object_box = None
        
        if hasattr(self, '_last_hand_box') and self._last_hand_box is not None:
            current_hand_box = self._last_hand_box.copy()
        
        if hasattr(self, '_last_object_box') and self._last_object_box is not None:
            current_object_box = self._last_object_box.copy()
        
        # **FIXED: Use only reset_state for memory management (no redundant init_state)**
        # reset_state clears memory while preserving the inference state structure
        self._sam2_video.reset_state(self._inference_state)
        
        # Clear the cached images tensor and reset frame count
        self._inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        self._inference_state["num_frames"] = 0
        
        # Set video dimensions and add current frame
        self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
        frame_idx = self._sam2_video.add_new_frame(self._inference_state, frame_rgb)
        
        # Re-prime with preserved tracking state
        reprime_success = False
        
        if current_hand_box is not None and self._tracked_hand_id is not None:
            try:
                _, out_obj_ids_hand, out_mask_logits_hand = self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_hand_id,
                    box=current_hand_box,
                )
                reprime_success = True
            except Exception as e:
                print(f"⚠️  Failed to re-prime hand tracking: {e}")
                self._tracked_hand_id = None
        
        if current_object_box is not None and self._tracked_object_id is not None:
            try:
                _, out_obj_ids_object, out_mask_logits_object = self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_object_id,
                    box=current_object_box,
                )
                reprime_success = True
            except Exception as e:
                print(f"⚠️  Failed to re-prime object tracking: {e}")
                self._tracked_object_id = None
        
        # Update frame count and priming status
        self._frame_count = frame_idx + 1
        self._sam2_primed = reprime_success
        
        if not reprime_success:
            # Reset tracking IDs if re-priming failed
            self._tracked_object_id = None
            self._tracked_hand_id = None
        
        return True
    
    def start_hand_tracking(self, frame_rgb: np.ndarray) -> bool:
        """Start tracking the user's hand for hand-first workflow."""
        try:
            print("🤚 Starting hand detection...")
            
            hand_prompt = f"my {self.handedness} hand"
            
            # Use unified detection method
            success = self._detect_and_prime_sam2(frame_rgb, hand_prompt, ["hand"])
            
            if success:
                self._hand_initialized = True
                
                # Update state machine to reflect successful hand tracking
                self.tracking_state_machine._transition_to(TrackingState.HAND_READY)
                
                # Update loss tracker with successful hand detection
                self.loss_tracker.update(False, True)  # No object, yes hand
                
                print(f"🤚 Hand tracking started! Ready for object prompt.")
                return True
            else:
                print(f"⚠️  No suitable hand found")
                return False
            
        except Exception as e:
            print(f"❌ Hand tracking initialization failed: {e}")
            return False
    
    def set_object_prompt(self, text: str) -> bool:
        """Set the object prompt for hand-first workflow.
        
        This method should be called after start_hand_tracking() is successful.
        It will begin searching for the specified object while continuing to track the hand.
        
        Args:
            text: Object prompt (e.g., "coffee cup")
            
        Returns:
            bool: True if object search started successfully
        """
        if not self._hand_initialized:
            print("⚠️  Hand tracking not initialized. Call start_hand_tracking() first.")
            return False
        
        if not self.tracking_state_machine.is_ready_for_object_prompt():
            print(f"⚠️  System not ready for object prompt. Current state: {self.tracking_state_machine.state.value}")
            return False
        
        self._prompt = text
        self._object_prompt_provided = True
        
        # Transition state machine to object search
        success = self.tracking_state_machine.start_object_search()
        if success:
            print(f"🔍 Starting search for '{text}'...")
            return True
        else:
            print("❌ Failed to start object search")
            return False
    
    def is_ready_for_object_prompt(self) -> bool:
        """Check if system is ready to receive object prompt."""
        return (self._hand_initialized and 
                self.tracking_state_machine.is_ready_for_object_prompt())
    
    def get_status_message(self) -> str:
        """Get current status message for user feedback."""
        state = self.tracking_state_machine.state
        
        if state == TrackingState.WAITING_FOR_HAND:
            return f"Looking for hand... (frame {self._search_frame_counter}/{self._search_interval})"
        elif state == TrackingState.HAND_READY:
            return "Hand detected! Ready for object prompt."
        elif state == TrackingState.SEARCHING_OBJECT:
            attempts = self.tracking_state_machine.object_search_attempts
            max_attempts = self.tracking_state_machine.max_object_search_attempts
            return f"Searching for '{self._prompt}'... ({attempts}/{max_attempts}, frame {self._search_frame_counter}/{self._search_interval})"
        elif state == TrackingState.TRACKING_BOTH:
            return f"Tracking hand and '{self._prompt}'"
        else:
            return f"Status: {state.value}" 

    def _attempt_object_detection(self, frame_bgr: np.ndarray) -> bool:
        """Detect object and add it to SAM-2 using unified detection logic."""
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Use unified detection with state reset and hand preservation
            success = self._detect_and_prime_sam2(
                frame_rgb, 
                self._prompt, 
                ["object"], 
                reset_state=True, 
                preserve_existing=["hand"]
            )
            
            return success
        
        except Exception as e:
            print(f"❌ Object detection failed: {e}")
            return False
    
    def _add_frame_streaming(self, inference_state: dict, frame_rgb: np.ndarray) -> int:
        """Ultra-fast frame addition that bypasses SAM-2's slow PIL preprocessing.
        
        This manually implements the essential steps of add_new_frame but with
        OpenCV instead of PIL LANCZOS, achieving 5-10x speedup.
        
        Args:
            inference_state: SAM-2 inference state
            frame_rgb: RGB frame as numpy array
            
        Returns:
            int: Frame index
        """
        device = inference_state["device"]
        image_size = 1024  # SAM-2 standard size
        
        # Step 1: Fast OpenCV preprocessing (instead of slow PIL LANCZOS)
        frame_resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        frame_float = frame_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).to(device)
        
        # Step 2: ImageNet normalization
        img_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)[:, None, None]
        img_tensor.sub_(img_mean).div_(img_std)
        
        # Step 3: Handle image sequence initialization/concatenation
        images = inference_state.get("images", None)
        if images is None or (isinstance(images, list) and len(images) == 0):
            inference_state["images"] = img_tensor.unsqueeze(0)
        else:
            img_tensor = img_tensor.to(images.device)
            inference_state["images"] = torch.cat([images, img_tensor.unsqueeze(0)], dim=0)
        
        # Step 4: Update frame count
        inference_state["num_frames"] = inference_state["images"].shape[0]
        frame_idx = inference_state["num_frames"] - 1
        
        # Step 5: Cache visual features
        image_batch = img_tensor.float().unsqueeze(0)
        backbone_out = self._sam2_video.forward_image(image_batch)
        inference_state["cached_features"][frame_idx] = (image_batch, backbone_out)
        
        return frame_idx 

    def get_performance_stats(self) -> dict:
        """Get performance statistics for the tracking system."""
        current_time = time.time()
        elapsed_time = current_time - self._start_time
        average_fps = self._total_frames_processed / elapsed_time if elapsed_time > 0 else 0
        return {
            "total_frames_processed": self._total_frames_processed,
            "elapsed_time": elapsed_time,
            "average_fps": average_fps,
        }
    
    def print_performance_summary(self) -> None:
        """Print a simple performance summary."""
        stats = self.get_performance_stats()
        print(f"\n📊 GSAM2 Performance Summary:")
        print(f"   Total frames processed: {stats['total_frames_processed']}")
        print(f"   Total time: {stats['elapsed_time']:.2f}s")
        print(f"   Average FPS: {stats['average_fps']:.1f}")
        print(f"   Real-time capable: {'✅ YES' if stats['average_fps'] >= 25 else '❌ NO'} (25+ FPS)")
        print() 