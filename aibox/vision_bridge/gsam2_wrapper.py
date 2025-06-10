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

# **Performance Timing**
from collections import defaultdict
from contextlib import contextmanager

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
    RECOVERY = "recovery"                       # Lost tracking, attempting recovery


class SimpleLossTracker:
    """Simple tracking loss detection - only what we actually use."""
    
    def __init__(self, max_loss_frames: int = 8):
        self.max_loss_frames = max_loss_frames  # Consider lost after 8 consecutive missed frames (~0.3s at 30fps)
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
    """Simplified state machine - only essential states and transitions."""
    
    def __init__(self, loss_tracker: SimpleLossTracker, hand_first_mode: bool = True):
        self.hand_first_mode = hand_first_mode
        self.loss_tracker = loss_tracker
        
        # **SIMPLIFIED: Only two initial states**
        if hand_first_mode:
            self.state = TrackingState.WAITING_FOR_HAND
        else:
            self.state = TrackingState.TRACKING_BOTH  # Legacy mode starts in tracking
            
        self.state_entry_time = time.time()
        self.object_search_attempts = 0
        self.max_object_search_attempts = 10  # ~5s at 30fps
        
    def update_state(self, has_object: bool, has_hand: bool) -> TrackingState:
        """Simplified state machine with only essential transitions."""
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
        
        elif self.state == TrackingState.RECOVERY:
            if has_hand and has_object:
                self._transition_to(TrackingState.TRACKING_BOTH)
            elif has_hand:
                self._transition_to(TrackingState.HAND_READY)
            else:
                self._transition_to(TrackingState.WAITING_FOR_HAND)
        
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
        if self.hand_first_mode:
            self.state = TrackingState.WAITING_FOR_HAND
        else:
            self.state = TrackingState.TRACKING_BOTH
            
        self.state_entry_time = time.time()
        self.object_search_attempts = 0


class PerformanceTimer:
    """Simple performance timer for measuring FPS and component timing."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timings = defaultdict(list)
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    @contextmanager
    def time_component(self, component_name: str):
        """Context manager for timing specific components."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[component_name].append(elapsed)
            # Keep only recent measurements
            if len(self.timings[component_name]) > self.window_size:
                self.timings[component_name] = self.timings[component_name][-self.window_size:]
    
    def frame_tick(self):
        """Call this at the end of each frame to update FPS."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        self.frame_count += 1
    
    def get_fps(self) -> float:
        """Get current FPS based on recent frames."""
        if len(self.frame_times) < 2:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_component_stats(self) -> dict:
        """Get timing statistics for all components."""
        stats = {}
        for component, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                stats[component] = {
                    'avg_ms': avg_time * 1000,
                    'max_ms': max_time * 1000,
                    'min_ms': min_time * 1000,
                    'fps_if_bottleneck': 1.0 / avg_time if avg_time > 0 else 0.0
                }
        return stats
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary."""
        fps = self.get_fps()
        stats = self.get_component_stats()
        
        print(f"\n📊 Performance Summary (last {len(self.frame_times)} frames):")
        print(f"   Overall FPS: {fps:.1f}")
        print(f"   Total frames processed: {self.frame_count}")
        
        if stats:
            print(f"   Component breakdown:")
            for component, timing in stats.items():
                print(f"     {component}: {timing['avg_ms']:.1f}ms avg ({timing['fps_if_bottleneck']:.1f} FPS if bottleneck)")
        print()


class GSAM2Wrapper:
    """A *minimal* wrapper that exposes a YOLO-compatible interface
    around Grounding-DINO + SAM-2 tracking.

    Uses Grounding-DINO for initial object detection and SAM-2 for 
    temporal tracking with mask propagation. DINO is only used for
    detection/re-priming - never as a per-frame fallback tracker.
    Converts masks back to bounding boxes to maintain compatibility 
    with bracelet navigation.

    Public API
    ----------
    set_prompt(frame_rgb, text)
        Initialize SAM-2 tracking with first detection from text prompt.
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
        memory_monitoring: bool = True,
        hand_first_mode: bool = True,  # New: Enable hand-first workflow
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.handedness = handedness
        self.frame_cache_limit = frame_cache_limit
        self.memory_monitoring = memory_monitoring
        self.hand_first_mode = hand_first_mode

        # Load Grounding-DINO for detection
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
        
        # **DEBUG: Check SAM-2 attention backend settings**
        old_gpu, use_flash_attn, math_kernel_on = get_sdpa_settings()
        print(f"🔧 SAM-2 Attention Backend Settings:")
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"   - GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"   - GPU Name: {gpu_props.name}")
        else:
            print(f"   - GPU: Not available (CPU mode)")
            
        print(f"   - Flash Attention Enabled: {use_flash_attn}")
        print(f"   - Math Kernel Enabled: {math_kernel_on}")
        print(f"   - Old GPU Mode: {old_gpu}")
        print(f"   - PyTorch Version: {torch.__version__}")
        
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
        self.tracking_state_machine = TrackingStateMachine(self.loss_tracker, hand_first_mode)

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

        # Initialize performance timer
        self.performance_timer = PerformanceTimer()
        
        # **NEW: GDINO call counter for efficiency tracking**
        self._gdino_call_count = 0
        self._gdino_hand_calls = 0
        self._gdino_object_calls = 0
        
        # **NEW: Separate SAM-2 tracking FPS (excluding GDINO spikes)**
        self._sam2_frame_times = deque(maxlen=30)
        self._sam2_frame_count = 0
        
        # **NEW: Steady-state tracking metrics (after initial setup)**
        self._setup_complete = False
        self._steady_state_frame_times = deque(maxlen=60)  # Longer window for steady state
        self._steady_state_frame_count = 0
        self._memory_reset_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_prompt(self, frame_rgb: np.ndarray | None, text: str) -> None:
        """(Re)sets the text prompt and handles SAM-2 tracking transitions.
        
        For hand-first workflow, this is equivalent to set_object_prompt().
        For legacy workflow, performs full initialization or mid-tracking changes.

        Parameters
        ----------
        frame_rgb : np.ndarray | None
            RGB frame for object detection and SAM-2 priming. If None, only updates
            the text prompt without priming.
        text : str
            The free-form text prompt (e.g. "red bottle").
        """
        
        # Handle hand-first workflow
        if self.hand_first_mode:
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
            return
        
        # Legacy workflow (existing behavior)
        old_prompt = self._prompt
        self._prompt = text
        
        # If no frame provided, just update the prompt
        if frame_rgb is None:
            print(f"🔄 Prompt updated: '{old_prompt}' → '{text}' (no frame provided)")
            return
        
        # Check if this is a mid-tracking prompt change (SAM-2 already primed)
        if self._sam2_primed and old_prompt != text:
            print(f"🔄 Mid-tracking prompt change: '{old_prompt}' → '{text}'")
            self._handle_prompt_change_during_tracking(frame_rgb)
        else:
            # Initial setup or same prompt - do full initialization
            if not self._sam2_primed:
                print(f"🎯 Initial SAM-2 setup with prompt: '{text}'")
            
            # Reset SAM-2 inference state for new prompt
            self._inference_state = self._sam2_video.init_state(video_path=None)
            self._inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
            self._frame_count = 0

            # Reset loss tracker and state machine
            self.loss_tracker.reset()
            self.tracking_state_machine.reset()
            
            # Initialize tracking state
            self._sam2_primed = False
            self._tracked_object_id = None
            self._tracked_hand_id = None
            
            # **NEW: Reset frame counters on prompt change**
            self._search_frame_counter = 0
            
            # Perform initial detection and priming
            self._prime_sam2_with_detection(frame_rgb)
    
    def _handle_prompt_change_during_tracking(self, frame_rgb: np.ndarray) -> None:
        """Handle prompt changes during active tracking by preserving hand tracking.
        
        Strategy:
        1. Extract current hand mask from SAM-2 if being tracked
        2. Detect new object with Grounding-DINO
        3. Re-prime SAM-2 with preserved hand + new object
        
        Args:
            frame_rgb: Current RGB frame for new object detection
        """
        try:
            # Step 1: Preserve current hand tracking if exists
            preserved_hand_mask = None
            preserved_hand_box = None
            
            if self._tracked_hand_id is not None:
                # Get current hand mask from SAM-2
                try:
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    sam2_detections = self._track_with_sam2(frame_bgr, None)
                    
                    # Find current hand detection
                    for det in sam2_detections:
                        if len(det) >= 6 and det[5] == 1:  # class_id=1 for hand
                            xc, yc, w, h = det[:4]
                            preserved_hand_box = np.array([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
                            print(f"🤚 Preserving hand tracking: box={preserved_hand_box}")
                            break
                            
                except Exception as e:
                    print(f"⚠️  Could not preserve hand tracking: {e}")
            
            # Step 2: Detect new object with Grounding-DINO
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Only detect the new object (not hand, since we're preserving it)
            detections, labels = self._call_gdino_with_tracking(
                frame_bgr=frame_bgr,
                caption=self._prompt,  # Only new object prompt
                call_type="object"
            )
            
            if len(detections.xyxy) == 0:
                print(f"⚠️  No objects detected for new prompt: '{self._prompt}'")
                return
            
            # Select best object candidate
            object_candidates = []
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i])
                label = labels[i].lower() if i < len(labels) else ""
                object_candidates.append((x1, y1, x2, y2, conf, label))
            
            best_object = self._select_best_candidate(object_candidates, "object")
            if best_object is None:
                print(f"⚠️  No suitable objects found for prompt: '{self._prompt}'")
                return
            
            # Step 3: Reset SAM-2 and re-prime with preserved hand + new object
            self._inference_state = self._sam2_video.init_state(video_path=None)
            self._inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
            self._frame_count = 0
            
            # Set video dimensions
            self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
            
            # Add current frame to SAM-2
            frame_idx = self._sam2_video.add_new_frame(self._inference_state, frame_rgb)
            
            # Prime with new object
            x1, y1, x2, y2, conf, label = best_object
            object_box = np.array([x1, y1, x2, y2])
            
            self._tracked_object_id = 1  # Object gets ID 1
            _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=self._tracked_object_id,
                box=object_box,
            )
            print(f"🎯 Re-primed SAM-2 with new object '{label}' (ID: {self._tracked_object_id}, conf: {conf:.3f})")
            
            # Prime with preserved hand if available
            if preserved_hand_box is not None:
                self._tracked_hand_id = 2  # Hand gets ID 2
                _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_hand_id,
                    box=preserved_hand_box,
                )
                print(f"🤚 Re-primed SAM-2 with preserved hand (ID: {self._tracked_hand_id})")
            else:
                # Try to detect hand if not preserved
                hand_prompt = f"my {self.handedness} hand"
                hand_detections, hand_labels = self._call_gdino_with_tracking(
                    frame_bgr=frame_bgr,
                    caption=hand_prompt,
                    call_type="hand"
                )
                
                if len(hand_detections.xyxy) > 0:
                    # Find best hand candidate
                    hand_candidates = []
                    for i in range(len(hand_detections.xyxy)):
                        x1, y1, x2, y2 = hand_detections.xyxy[i]
                        conf = float(hand_detections.confidence[i])
                        label = hand_labels[i].lower() if i < len(hand_labels) else ""
                        if "hand" in label:
                            hand_candidates.append((x1, y1, x2, y2, conf, label))
                    
                    best_hand = self._select_best_candidate(hand_candidates, "hand")
                    if best_hand is not None:
                        x1, y1, x2, y2, conf, label = best_hand
                        hand_box = np.array([x1, y1, x2, y2])
                        
                        self._tracked_hand_id = 2
                        _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                            inference_state=self._inference_state,
                            frame_idx=frame_idx,
                            obj_id=self._tracked_hand_id,
                            box=hand_box,
                        )
                        print(f"🤚 Re-detected and primed hand '{label}' (ID: {self._tracked_hand_id}, conf: {conf:.3f})")
            
            self._frame_count = frame_idx + 1
            print(f"✅ Prompt change completed successfully")
            
            # Reset loss tracker but keep state machine in tracking mode
            self.loss_tracker.reset()
            # Don't reset state machine - we want to continue tracking
            
        except Exception as e:
            print(f"❌ Prompt change failed: {e}")
            # Fallback to full re-initialization
            print("🔄 Falling back to full re-initialization...")
            self._inference_state = self._sam2_video.init_state(video_path=None)
            self._inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
            self._frame_count = 0
            self.loss_tracker.reset()
            self.tracking_state_machine.reset()
            self._sam2_primed = False
            self._tracked_object_id = None
            self._tracked_hand_id = None
            self._prime_sam2_with_detection(frame_rgb)
    
    def _prime_sam2_with_detection(self, frame_rgb: np.ndarray) -> bool:
        """Prime SAM-2 with object and hand detection from first frame.
        
        Args:
            frame_rgb: RGB frame for object detection
            
        Returns:
            bool: True if priming successful, False otherwise
        """
        try:
            # Convert RGB to BGR for Grounding-DINO
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Multi-prompt: detect both target object and hand
            multi_prompt = f"{self._prompt}. my {self.handedness} hand"
            
            # Grounding-DINO detection for both object and hand
            with self.performance_timer.time_component("gdino_detection"):
                detections, labels = self._call_gdino_with_tracking(
                    frame_bgr=frame_bgr,
                    caption=multi_prompt,
                    call_type="recovery"
                )
            
            if len(detections.xyxy) == 0:
                print(f"⚠️  No objects detected for prompt: '{multi_prompt}'")
                return False
            
            # Separate object and hand detections
            object_candidates = []
            hand_candidates = []
            
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i])
                label = labels[i].lower() if i < len(labels) else ""
                
                detection_data = (x1, y1, x2, y2, conf, label)
                
                if "hand" in label:
                    hand_candidates.append(detection_data)
                else:
                    object_candidates.append(detection_data)
            
            # Select best candidates
            best_object = self._select_best_candidate(object_candidates, "object")
            best_hand = self._select_best_candidate(hand_candidates, "hand")
            
            if best_object is None and best_hand is None:
                print(f"⚠️  No suitable objects found for prompt: '{multi_prompt}'")
                return False
            
            # **CRITICAL**: Set video dimensions in inference state FIRST
            self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
            
            # Add RGB frame to SAM-2 inference state
            with self.performance_timer.time_component("sam2_add_frame"):
                frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
            
            # Prime SAM-2 with detected objects
            primed_objects = []
            
            if best_object is not None:
                x1, y1, x2, y2, conf, label = best_object
                object_box = np.array([x1, y1, x2, y2])
                
                self._tracked_object_id = 1  # Object gets ID 1
                
                with self.performance_timer.time_component("sam2_prime_object"):
                    print(f"🔧 SAM-2 Re-priming: Adding object box with attention backend")
                    _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                        inference_state=self._inference_state,
                        frame_idx=frame_idx,
                        obj_id=self._tracked_object_id,
                        box=object_box,
                    )
                
                primed_objects.append(f"object '{label}' (ID: {self._tracked_object_id}, conf: {conf:.3f})")
                print(f"🎯 Primed SAM-2 with {primed_objects[-1]}")
            
            if best_hand is not None:
                x1, y1, x2, y2, conf, label = best_hand
                hand_box = np.array([x1, y1, x2, y2])
                
                self._tracked_hand_id = 2  # Hand gets ID 2
                
                with self.performance_timer.time_component("sam2_prime_hand"):
                    print(f"🔧 SAM-2 Re-priming: Adding hand box with attention backend")
                    _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                        inference_state=self._inference_state,
                        frame_idx=frame_idx,
                        obj_id=self._tracked_hand_id,
                        box=hand_box,
                    )
                
                primed_objects.append(f"hand '{label}' (ID: {self._tracked_hand_id}, conf: {conf:.3f})")
                print(f"🤚 Primed SAM-2 with {primed_objects[-1]}")
            
            self._sam2_primed = True
            self._frame_count = frame_idx + 1
            print(f"✅ SAM-2 primed successfully with {len(primed_objects)} objects at frame {frame_idx}")
            return True
            
        except Exception as e:
            print(f"❌ SAM-2 priming failed: {e}")
            return False
    
    def _select_best_candidate(self, candidates: List[tuple], candidate_type: str) -> Optional[tuple]:
        """Simplified candidate selection - just pick highest confidence."""
        if not candidates:
            return None
        
        # Simple confidence-based selection
        best_candidate = max(candidates, key=lambda x: x[4])  # x[4] is confidence
        
        return best_candidate

    def _call_gdino_with_tracking(self, frame_bgr: np.ndarray, caption: str, call_type: str = "unknown") -> tuple:
        """Call GDINO with usage tracking for efficiency analysis.
        
        Args:
            frame_bgr: BGR frame for detection
            caption: Text prompt for detection
            call_type: Type of call ("hand", "object", "multi") for tracking
            
        Returns:
            tuple: (detections, labels) from GDINO
        """
        self._gdino_call_count += 1
        
        if "hand" in call_type:
            self._gdino_hand_calls += 1
        elif "object" in call_type:
            self._gdino_object_calls += 1
            
        # Print counter every call for visibility during testing
        if self._gdino_call_count <= 3:  # Only print first few calls to avoid spam
            print(f"📊 GDINO Call #{self._gdino_call_count} ({call_type})")
        
        with self.performance_timer.time_component("gdino_detection"):
            detections, labels = self._gdino.predict_with_caption(
                image=frame_bgr,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        
        return detections, labels

    @torch.inference_mode()
    def track(
        self,
        frame_bgr: np.ndarray,
        depth_img: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Simplified tracking that handles both hand-first and legacy workflows.

        Returns
        -------
        list[np.ndarray]
            List of detection tuples: [xc, yc, w, h, track_id, class_id, conf, depth]
            class_id: 0 = target object, 1 = hand
        """
        with self.performance_timer.time_component("total_frame"):
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
                    if self._search_frame_counter == 0 or self._search_frame_counter >= self._search_interval:
                        print(f"🔍 Attempting object detection...")
                        self._search_frame_counter = 0  # Reset counter after detection attempt
                        # Increment search attempt counter and check for timeout
                        if self.tracking_state_machine.increment_search_attempt():
                            detection_success = self._attempt_object_detection(frame_bgr)
                            if not detection_success:
                                # Increment counter if detection failed (to retry in 15 frames)
                                self._search_frame_counter += 1
                            else:
                                # Increment counter if detection succeeded (to avoid immediate retry)
                                self._search_frame_counter += 1
                        else:
                            print("🔍 Search timeout reached, stopping attempts")
                    else:
                        # Count frames between detection attempts
                        self._search_frame_counter += 1
                else:
                    print(f"✅ State transitioned to {current_state_after_sam2.value}, stopping object search")
            
            elif current_state == TrackingState.TRACKING_BOTH:
                results = self._track_with_sam2(frame_bgr, depth_img)
            
            elif current_state == TrackingState.RECOVERY:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if self._attempt_sam2_recovery(frame_rgb):
                    results = self._track_with_sam2(frame_bgr, depth_img)
                else:
                    self.tracking_state_machine._transition_to(TrackingState.WAITING_FOR_HAND)
                    results = []
            
            else:
                # Fallback
                results = self._track_with_sam2(frame_bgr, depth_img) if self._sam2_primed else []
            
            # Update performance timer
            self.performance_timer.frame_tick()
            
            # **NEW: Track steady-state performance (after initial setup)**
            if not self._setup_complete and self._gdino_call_count >= 2:
                # Setup complete after hand + object detection
                self._setup_complete = True
                print("🚀 Setup complete! Now tracking steady-state performance...")
            
            if self._setup_complete:
                # Record steady-state frame time (excludes setup GDINO calls)
                current_time = time.time()
                if hasattr(self, '_last_steady_frame_time'):
                    steady_frame_time = current_time - self._last_steady_frame_time
                    self._steady_state_frame_times.append(steady_frame_time)
                    self._steady_state_frame_count += 1
                self._last_steady_frame_time = current_time
            
            # Print performance summary every 60 frames (less frequent)
            if self.performance_timer.frame_count % 120 == 0:
                self.print_streamlined_performance_summary()
            
            return results
    
    def _handle_hand_initialization(self, frame_bgr: np.ndarray, depth_img: Optional[np.ndarray]) -> List[np.ndarray]:
        """Handle hand initialization state with frame-based detection intervals."""
        
        if not self._sam2_primed:
            # Attempt hand detection immediately on first frame (counter=0), then every 15 frames  
            if self._search_frame_counter == 0 or self._search_frame_counter >= self._search_interval:
                print(f"🤚 Attempting hand detection...")
                self._search_frame_counter = 0  # Reset counter after detection attempt
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if self.start_hand_tracking(frame_rgb):
                    # Increment counter if detection succeeded (to avoid immediate retry)
                    self._search_frame_counter += 1
                    return self._track_with_sam2(frame_bgr, depth_img)
                else:
                    # Increment counter if detection failed (to retry in 15 frames)
                    self._search_frame_counter += 1
                    return []
            else:
                # Count frames between detection attempts
                self._search_frame_counter += 1
                return []
        else:
            # SAM-2 is primed, continue tracking
            return self._track_with_sam2(frame_bgr, depth_img)
    
    def _track_with_sam2(self, frame_bgr: np.ndarray, depth_img: Optional[np.ndarray]) -> List[np.ndarray]:
        """Track using SAM-2 temporal tracking."""
        sam2_start_time = time.time()  # Track pure SAM-2 performance
        
        try:
            with self.performance_timer.time_component("sam2_tracking"):
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
                with self.performance_timer.time_component("sam2_add_frame"):
                    frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
                
                # Run SAM-2 inference on the frame
                with self.performance_timer.time_component("sam2_inference"):
                    # Removed verbose debug print - only print on first few frames or errors
                    frame_idx, tracked_obj_ids, video_res_masks = self._sam2_video.infer_single_frame(
                        inference_state=self._inference_state,
                        frame_idx=frame_idx,
                    )
                
                results = []
                
                # Process tracked masks and convert to Detection tuples
                with self.performance_timer.time_component("mask_to_bbox"):
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
                
                # **NEW: Record pure SAM-2 tracking performance**
                sam2_frame_time = time.time() - sam2_start_time
                self._sam2_frame_times.append(sam2_frame_time)
                self._sam2_frame_count += 1
                
                return results
                
        except Exception as e:
            print(f"⚠️  SAM-2 tracking failed: {e}")
            # Attempt recovery instead of falling back to DINO every frame
            # This preserves the efficiency benefits of SAM-2 vs DINO-per-frame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self._attempt_sam2_recovery(frame_rgb):
                print("✅ SAM-2 recovery successful, will retry next frame")
            else:
                print("❌ SAM-2 recovery failed - returning empty results")
                # State machine will handle transitions based on empty results
            return []
    
    def _filter_egocentric_hands(self, detections: List[np.ndarray], frame_shape: tuple) -> List[np.ndarray]:
        """Filter to most likely egocentric hand based on spatial rules and handedness.
        
        Args:
            detections: List of hand detection arrays
            frame_shape: (height, width, channels) of the frame
            
        Returns:
            Filtered list of hand detections
        """
        # Placeholder implementation - return all detections as-is
        return detections 
    
    # ------------------------------------------------------------------
    # Memory Management
    # ------------------------------------------------------------------
    
    def _check_memory_reset(self, frame_rgb: np.ndarray) -> bool:
        """Simple memory management - use original approach until SAM-2 API is clarified."""
        if self._inference_state["images"].shape[0] >= self.frame_cache_limit:
            self._memory_reset_count += 1  # Track memory resets for performance analysis
            return self._fallback_full_reset(frame_rgb)
        return False
    
    def _fallback_full_reset(self, frame_rgb: np.ndarray) -> bool:
        """Fallback full reset with tracking preservation (previous approach)."""
        self._memory_reset_count += 1  # Track memory resets for performance analysis
        
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
    
    def _attempt_sam2_recovery(self, frame_rgb: np.ndarray) -> bool:
        """Attempt to recover lost SAM-2 tracking by re-priming with fresh detection."""
        try:
            # Use Grounding-DINO to detect objects again
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            multi_prompt = f"{self._prompt}. my {self.handedness} hand"
            
            detections, labels = self._call_gdino_with_tracking(
                frame_bgr=frame_bgr,
                caption=multi_prompt,
                call_type="recovery"
            )
            
            if len(detections.xyxy) == 0:
                return False
            
            # Separate object and hand detections
            object_candidates = []
            hand_candidates = []
            
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i])
                label = labels[i].lower() if i < len(labels) else ""
                
                detection_data = (x1, y1, x2, y2, conf, label)
                
                if "hand" in label:
                    hand_candidates.append(detection_data)
                else:
                    object_candidates.append(detection_data)
            
            # Select best candidates
            best_object = self._select_best_candidate(object_candidates, "object")
            best_hand = self._select_best_candidate(hand_candidates, "hand")
            
            if best_object is None and best_hand is None:
                return False
            
            # Get current loss status to determine what needs recovery
            loss_status = self.loss_tracker.get_loss_status()
            
            # Re-prime SAM-2 with newly detected objects
            if self._inference_state["images"].shape[0] > 0:
                self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
                frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
            else:
                # Fresh start - reset inference state
                self._inference_state = self._sam2_video.init_state(video_path=None)
                self._inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
                self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
                frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
                self._frame_count = 0
            
            recovered_targets = []
            
            # Re-prime object if it was lost and now detected
            if best_object is not None and (loss_status["object_lost"] or self._tracked_object_id is None):
                x1, y1, x2, y2, conf, label = best_object
                object_box = np.array([x1, y1, x2, y2])
                
                self._tracked_object_id = 1  # Object gets ID 1
                
                _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_object_id,
                    box=object_box,
                )
                
                recovered_targets.append("object")
            
            # Re-prime hand if it was lost and now detected
            if best_hand is not None and (loss_status["hand_lost"] or self._tracked_hand_id is None):
                x1, y1, x2, y2, conf, label = best_hand
                hand_box = np.array([x1, y1, x2, y2])
                
                self._tracked_hand_id = 2  # Hand gets ID 2
                
                _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=frame_idx,
                    obj_id=self._tracked_hand_id,
                    box=hand_box,
                )
                
                recovered_targets.append("hand")
            
            if recovered_targets:
                self._sam2_primed = True
                self._frame_count = frame_idx + 1
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ SAM-2 recovery failed: {e}")
            return False

    def start_hand_tracking(self, frame_rgb: np.ndarray) -> bool:
        """Initialize hand-first tracking workflow.
        
        This method starts the hand-first workflow by detecting and tracking
        the user's hand. Once successful, the system will be ready to accept
        object prompts via set_object_prompt().
        
        Args:
            frame_rgb: RGB frame for hand detection
            
        Returns:
            bool: True if hand tracking started successfully
        """
        if not self.hand_first_mode:
            print("⚠️  Hand-first mode not enabled")
            return False
            
        try:
            print("🤚 Starting hand detection...")
            
            # Convert RGB to BGR for Grounding-DINO
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Hand-only prompt
            hand_prompt = f"my {self.handedness} hand"
            
            # Grounding-DINO detection for hand only
            detections, labels = self._call_gdino_with_tracking(
                frame_bgr=frame_bgr,
                caption=hand_prompt,
                call_type="hand"
            )
            
            if len(detections.xyxy) == 0:
                print(f"⚠️  No hand detected")
                return False
            
            # Find best hand candidate
            hand_candidates = []
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i])
                label = labels[i].lower() if i < len(labels) else ""
                
                if "hand" in label:
                    hand_candidates.append((x1, y1, x2, y2, conf, label))
            
            best_hand = self._select_best_candidate(hand_candidates, "hand")
            if best_hand is None:
                print(f"⚠️  No suitable hand found")
                return False
            
            # Set video dimensions in inference state
            self._inference_state["video_height"], self._inference_state["video_width"] = frame_rgb.shape[:2]
            
            # Add RGB frame to SAM-2 inference state
            frame_idx = self._add_frame_streaming(self._inference_state, frame_rgb)
            
            # Prime SAM-2 with hand detection
            x1, y1, x2, y2, conf, label = best_hand
            hand_box = np.array([x1, y1, x2, y2])
            
            self._tracked_hand_id = 2  # Hand gets ID 2
            
            _, out_obj_ids, out_mask_logits = self._sam2_video.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=self._tracked_hand_id,
                box=hand_box,
            )
            
            self._sam2_primed = True
            self._hand_initialized = True
            self._frame_count = frame_idx + 1
            
            # Update state machine to reflect successful hand tracking
            self.tracking_state_machine._transition_to(TrackingState.HAND_READY)
            
            # Update loss tracker with successful hand detection
            hand_detection = np.array([
                (x1 + x2) / 2, (y1 + y2) / 2,  # xc, yc
                x2 - x1, y2 - y1,              # w, h
                self._tracked_hand_id, 1, conf, -1  # track_id, class_id, conf, depth
            ])
            self.loss_tracker.update(False, True)  # No object, yes hand
            
            print(f"🤚 Hand tracking started! Ready for object prompt.")
            
            return True
            
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
        if not self.hand_first_mode:
            print("⚠️  Hand-first mode not enabled")
            return False
            
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
        return (self.hand_first_mode and 
                self._hand_initialized and 
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
        elif state == TrackingState.RECOVERY:
            return "Attempting recovery..."
        else:
            return f"Status: {state.value}" 

    def _attempt_object_detection(self, frame_bgr: np.ndarray) -> bool:
        """Detect object and properly add it to SAM-2 using reset_state approach."""
        try:
            with self.performance_timer.time_component("object_detection_attempt"):
                with self.performance_timer.time_component("gdino_object_detection"):
                    detections, labels = self._call_gdino_with_tracking(
                        frame_bgr=frame_bgr,
                        caption=self._prompt,
                        call_type="object"
                    )

                if len(detections.xyxy) == 0:
                    return False
                
                # Take first decent detection
                x1, y1, x2, y2 = detections.xyxy[0]
                conf = float(detections.confidence[0])
                object_box = np.array([x1, y1, x2, y2])
                
                # Use proper SAM-2 workflow to add object during tracking
                success = self._add_object_with_reset(frame_bgr, object_box)
                return success
            
        except Exception as e:
            print(f"❌ Object detection failed: {e}")
            return False
    
    def _add_object_with_reset(self, frame_bgr: np.ndarray, object_box: np.ndarray) -> bool:
        """Add object using proper SAM-2 workflow with reset_state (official approach)."""
        try:
            # Convert BGR to RGB for SAM-2
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Get current hand bounding box from SAM-2's tracking state
            if self._last_hand_box is None:
                return False
            
            hand_box = self._last_hand_box.copy()
            
            # Reset SAM-2 state and re-add current frame
            self._sam2_video.reset_state(self._inference_state)
            frame_idx = self._sam2_video.add_new_frame(self._inference_state, frame_rgb)
            
            # Add hand first (preserve existing tracking)
            self._tracked_hand_id = 2  # Hand gets ID 2
            _, out_obj_ids_hand, out_mask_logits_hand = self._sam2_video.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=self._tracked_hand_id,
                box=hand_box,
            )
            
            # Add new object
            self._tracked_object_id = 1  # Object gets ID 1
            _, out_obj_ids_object, out_mask_logits_object = self._sam2_video.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=self._tracked_object_id,
                box=object_box,
            )
            
            # Update frame count and state
            self._frame_count = frame_idx + 1
            self._last_object_box = object_box.copy()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add object with reset: {e}")
            return False

    def get_performance_stats(self) -> dict:
        """Get current performance statistics.
        
        Returns:
            dict: Performance statistics including FPS and component timings
        """
        return {
            'fps': self.performance_timer.get_fps(),
            'frame_count': self.performance_timer.frame_count,
            'component_stats': self.performance_timer.get_component_stats()
        }
    
    def print_streamlined_performance_summary(self):
        """Print streamlined performance focused on steady-state tracking efficiency."""
        overall_fps = self.performance_timer.get_fps()
        
        # Calculate steady-state FPS (after initial setup)
        steady_state_fps = 0.0
        if len(self._steady_state_frame_times) > 0:
            avg_steady_time = sum(self._steady_state_frame_times) / len(self._steady_state_frame_times)
            steady_state_fps = 1.0 / avg_steady_time if avg_steady_time > 0 else 0.0
        
        # Calculate pure SAM-2 tracking performance
        sam2_fps = 0.0
        if len(self._sam2_frame_times) > 0:
            sam2_avg_time = sum(self._sam2_frame_times) / len(self._sam2_frame_times)
            sam2_fps = 1.0 / sam2_avg_time if sam2_avg_time > 0 else 0.0
        
        # Get key component timings (only the essential ones)
        component_stats = self.performance_timer.get_component_stats()
        sam2_inference_ms = component_stats.get("sam2_inference", {}).get("avg_ms", 0)
        sam2_add_frame_ms = component_stats.get("sam2_add_frame", {}).get("avg_ms", 0)
        
        print(f"\n🎯 **TRACKING PERFORMANCE SUMMARY**")
        print(f"   📊 Setup Status: {'✅ Complete' if self._setup_complete else '🔄 In Progress'}")
        print(f"   📊 GDINO Calls: {self._gdino_call_count} total (hand={self._gdino_hand_calls}, object={self._gdino_object_calls})")
        print(f"   📊 Frame-to-GDINO Ratio: {max(1, self.performance_timer.frame_count) / max(1, self._gdino_call_count):.1f}:1")
        print(f"")
        print(f"   🚀 **STEADY-STATE TRACKING** (after setup):")
        print(f"      - Steady-State FPS: {steady_state_fps:.1f} (includes resets with preserved boxes)")
        print(f"      - Frames processed: {self._steady_state_frame_count}")
        print(f"      - Memory resets: {self._memory_reset_count} (using preserved boxes, no GDINO)")
        print(f"")
        print(f"   ⚡ **SAM-2 COMPONENT PERFORMANCE**:")
        print(f"      - SAM-2 Tracking FPS: {sam2_fps:.1f} (pure tracking, no resets)")
        print(f"      - Inference: {sam2_inference_ms:.1f}ms avg")
        print(f"      - Frame Addition: {sam2_add_frame_ms:.1f}ms avg")
        print(f"")
        print(f"   📈 **EFFICIENCY ANALYSIS**:")
        if self._setup_complete:
            efficiency = (steady_state_fps / sam2_fps * 100) if sam2_fps > 0 else 0
            print(f"      - Steady-State Efficiency: {efficiency:.1f}% of pure SAM-2 speed")
            print(f"      - Performance Impact: {sam2_fps - steady_state_fps:.1f} FPS lost to overhead")
        else:
            print(f"      - Waiting for setup completion to calculate efficiency...")
        print()
    
    def print_performance_summary(self):
        """Legacy method - redirect to streamlined version."""
        self.print_streamlined_performance_summary()

    def get_gdino_call_stats(self) -> dict:
        """Get GDINO call statistics for efficiency analysis.
        
        Returns:
            dict: GDINO call statistics
        """
        return {
            'total_calls': self._gdino_call_count,
            'hand_calls': self._gdino_hand_calls,
            'object_calls': self._gdino_object_calls,
            'other_calls': self._gdino_call_count - self._gdino_hand_calls - self._gdino_object_calls,
            'calls_per_frame': self._gdino_call_count / max(1, self.performance_timer.frame_count),
        }
    
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
        with self.performance_timer.time_component("fast_streaming"):
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
            with self.performance_timer.time_component("feature_extraction"):
                image_batch = img_tensor.float().unsqueeze(0)
                backbone_out = self._sam2_video.forward_image(image_batch)
                inference_state["cached_features"][frame_idx] = (image_batch, backbone_out)
            
            return frame_idx 