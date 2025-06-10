# GSAM2Wrapper Integration Documentation

## Overview

The **GSAM2Wrapper** provides an open-vocabulary, speech-driven object detection and tracking system that integrates seamlessly into the existing tactile guidance framework. It serves as a drop-in replacement for the traditional YOLOv5-based vision pipeline, enabling users to find objects using natural language descriptions rather than being limited to pre-trained object classes.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   master.py     │───▶│  controller.py   │───▶│ GSAM2Wrapper      │
│                 │    │  TaskController  │    │ (vision_bridge/)   │
│ • CLI args      │    │                  │    │                    │
│ • Setup         │    │ • Backend        │    │ • Grounding DINO   │
│ • Participant   │    │   selection      │    │ • SAM-2 tracking   │
│   management    │    │ • Frame loop     │    │ • Hand-first flow  │
└─────────────────┘    └──────────────────┘    └────────────────────┘
                                │                           │
                                │                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   bracelet.py   │◀───│  experiment_loop │    │    Detection       │ 
│                 │    │                  │    │    Tuples          │
│ • Navigation    │    │ • Frame          │    │                    │
│ • Haptic        │    │   processing     │    │ (xc, yc, w, h,     │
│   feedback      │    │ • Visualization  │    │  track_id,         │
│ • Motor control │    │ • Trial logic    │    │  class_id,         │
└─────────────────┘    └──────────────────┘    │  conf, depth)      │
                                                └────────────────────┘
```

## Data Flow

### 1. System Initialization (`master.py`)

```python
# Parse CLI arguments including backend selection
args = parser.parse_args()

# Initialize TaskController with backend choice
task_controller = controller.TaskController(
    backend=args.backend,  # "yolo" or "gsam2"
    prompt=args.prompt,    # Text prompt for GSAM2
    handedness=args.handedness,
    # ... other parameters
)
```

### 2. Vision Backend Setup (`controller.py::TaskController.__init__`)

```python
if self.backend == "gsam2":
    # Initialize GSAM2Wrapper with hand-first workflow
    self.gsam2 = GSAM2Wrapper(handedness=self.handedness)
    self.gsam2.set_prompt(None, self.prompt)
    # Create labels for visualization
    self.names_obj = {0: self.prompt}
else:
    # Traditional YOLO path
    self.gsam2 = None
```

### 3. Main Processing Loop (`experiment_loop`)

```python
for frame, (path, im, im0s, vid_cap, _) in enumerate(self.dataset):
    
    if self.backend == "gsam2":
        # **OPTIMIZED GSAM2 PATH**: Direct detection + tracking
        outputs = self.gsam2.track(im0)  # Returns Detection tuples
        
        # Skip YOLO preprocessing, inference, NMS, hand detection
        # GSAM2 handles everything internally
        
    else:
        # Traditional YOLO + StrongSORT pipeline
        # ... YOLO inference, NMS, tracking updates
    
    # Depth estimation (optional)
    if self.run_depth_estimator:
        depth_img, _ = self.depth_estimator.predict_depth(im0)
        outputs = bbs_to_depth(im0, depth_img, outputs)
    
    # Navigation using standardized Detection tuples
    grasped, curr_target = self.bracelet_controller.navigate_hand(
        self.belt_controller, outputs, self.class_target_obj, 
        hand_classes, depth_img, self.participant_vibration_intensities
    )
```

## Detection Tuple Format

Both YOLO and GSAM2 backends produce standardized detection tuples that the navigation system consumes:

```python
Detection = (
    xc: float,      # Center X coordinate (pixels)
    yc: float,      # Center Y coordinate (pixels)  
    w: float,       # Width (pixels)
    h: float,       # Height (pixels)
    track_id: int,  # Unique tracking ID (-1 if no tracking)
    class_id: int,  # Object class (0=target object, 1=hand)
    conf: float,    # Confidence score (0.0-1.0)
    depth: float    # Depth in meters (-1.0 if unavailable)
)
```

**Critical**: This format is consumed by `bracelet.py::navigate_hand()` and must remain consistent across all vision backends.

## Hand-First Workflow

The GSAM2Wrapper implements a **hand-first workflow** optimized for assistive technology:

### State Machine

```
WAITING_FOR_HAND ──────────────▶ HAND_READY
     ▲                               │
     │                               ▼
     │                         SEARCHING_OBJECT
     │                               │
     │                               ▼
     └─────────────────────────── TRACKING_BOTH
```

### State Descriptions

1. **WAITING_FOR_HAND**: System searches for user's hand every 15 frames
2. **HAND_READY**: Hand detected and tracked, ready for object prompt
3. **SEARCHING_OBJECT**: Hand tracked, actively searching for prompted object
4. **TRACKING_BOTH**: Both hand and object tracked, providing navigation

### Temporal Tracking

- **SAM-2 Memory**: Maintains tracking state across frames even when objects move out of view
- **Loss Tolerance**: 20 frames (~0.67s at 30fps) before considering tracking lost
- **Memory Management**: Automatic reset every 100 frames to prevent VRAM buildup

## Integration Points

### 1. CLI Interface (`master.py`)

```bash
# GSAM2 backend with custom prompt
python master.py --participant 1 --condition grasping \
    --backend gsam2 --prompt "red coffee mug" --handedness right

# Traditional YOLO backend  
python master.py --participant 1 --condition grasping \
    --backend yolo
```

### 2. Runtime Prompt Updates

Users can press **'p'** during execution to change the object prompt:

```python
# In experiment_loop visualization section
if self.backend == "gsam2" and pressed_key == ord('p'):
    new_prompt = input("Enter new text prompt: ")
    if new_prompt:
        self.gsam2.set_prompt(None, new_prompt)
        self.names_obj = {0: new_prompt}
```

### 3. Performance Optimizations

**GSAM2 Path Optimizations:**
- Skip YOLO tensor preprocessing entirely
- Reduce depth estimation frequency (every 20 frames vs 10)
- Pre-convert coordinates for visualization
- Bypass motion detection (SAM-2 handles temporal consistency)

**Memory Management:**
- Automatic SAM-2 state reset every 100 frames
- Preservation of tracking state during resets
- Efficient frame streaming without PIL overhead

### 4. Visualization Integration

```python
# Filename includes prompt for saved videos
if self.backend == "gsam2":
    clean_prompt = "".join(c if c.isalnum() else "_" for c in self.prompt)
    new_name = f"{p_stem}_{clean_prompt}_result{p_suffix}"
    save_path = str(save_dir / new_name)

# Unified visualization using Detection tuples
for xyxy, id, cls, conf, depth in visualization_data:
    label = f'ID: {id} {self.master_label[cls]} {conf:.2f} {depth:.2f}'
    annotator.box_label(xyxy, label, color=colors(cls, True))
```

## Performance Characteristics

### GSAM2 Backend
- **Latency**: ~40-50ms per frame (25+ FPS capable)
- **Memory**: ~3-4GB VRAM for SAM-2 + Grounding DINO
- **Accuracy**: Open-vocabulary detection with temporal consistency
- **Use Case**: Research, flexible object detection, assistive technology

### YOLO Backend  
- **Latency**: ~20-30ms per frame (30+ FPS)
- **Memory**: ~1-2GB VRAM
- **Accuracy**: Fixed object classes, proven detection
- **Use Case**: Production, known object sets, high-speed applications

## Configuration Options

### GSAM2Wrapper Parameters

```python
GSAM2Wrapper(
    device="cuda",              # GPU device
    box_threshold=0.35,         # GDINO detection threshold
    text_threshold=0.25,        # GDINO text threshold  
    default_prompt="coffee cup", # Initial object prompt
    handedness="right",         # User's dominant hand
    frame_cache_limit=100,      # SAM-2 memory reset interval
)
```

### Backend Selection

```python
# In TaskController.__init__
if self.backend == "gsam2":
    # Open-vocabulary pipeline
    self.gsam2 = GSAM2Wrapper(...)
    self.names_obj = {0: self.prompt}
else:
    # Traditional YOLO pipeline
    self.load_object_detector()
    self.load_object_tracker()
```

## Error Handling & Recovery

### Graceful Degradation
- **Detection Failures**: Return empty detection list, state machine handles transitions
- **SAM-2 Memory Issues**: Automatic reset with tracking preservation
- **Hand Tracking Loss**: Automatic return to hand detection mode
- **Object Tracking Loss**: Continue hand tracking, ready for new object prompt

### Debugging Features
- **Status Messages**: Real-time state and search progress
- **Performance Stats**: Frame processing rates and VRAM usage
- **Visual Feedback**: Bounding boxes, tracking IDs, confidence scores

## Usage Examples

### Basic Object Detection
```bash
python master.py --participant 0 --condition grasping \
    --backend gsam2 --prompt "water bottle"
```

### Multi-Object Session
```bash
python master.py --participant 1 --condition grasping \
    --backend gsam2 --prompt "coffee cup" --view
# Press 'p' during execution to change to "red apple", "smartphone", etc.
```

### Video Processing
```bash
python master.py --participant 0 --condition grasping \
    --backend gsam2 --source testingvid.mp4 --prompt "coffee mug"
```

## File Structure

```
tactile-guidance/aibox/
├── master.py                    # Entry point, CLI parsing
├── controller.py                # Main control logic, backend selection  
├── bracelet.py                  # Navigation logic (unchanged)
└── vision_bridge/
    ├── gsam2_wrapper.py         # GSAM2 integration wrapper
    └── README.md                # This documentation
```

## Integration Benefits

1. **Seamless Compatibility**: Drop-in replacement maintaining all existing interfaces
2. **Open Vocabulary**: Users can specify any object in natural language
3. **Temporal Consistency**: SAM-2 provides stable tracking across frames
4. **Hand-First Design**: Optimized workflow for assistive technology use cases
5. **Performance Optimized**: Streamlined processing pipeline with minimal overhead
6. **Robust Recovery**: Automatic error handling and state management

## Future Extensions

- **Speech-to-Text Integration**: Direct voice prompts using Whisper
- **Multi-Object Tracking**: Simultaneous tracking of multiple prompted objects
- **Gesture Recognition**: Hand pose analysis for interaction commands
- **Adaptive Prompting**: Learning user preferences and common object descriptions 