# EfficientTAM Integration with Grounded-SAM-2

This document describes the integration of **EfficientTAM** (Efficient Track Anything Model) with **Grounded-SAM-2**, allowing you to use EfficientTAM as a drop-in replacement for SAM2 models with improved efficiency.

## 🚀 Quick Start

### 1. Download EfficientTAM Checkpoints

```bash
# Download the tiny model (fastest, ~68MB)
cd checkpoints
wget https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_ti.pt

# Download the small model (balanced, ~130MB)
wget https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s.pt

# Or use the provided script to download all models
cd ..
bash download_efficienttam_ckpts.sh
```

### 2. Use EfficientTAM in Existing Demos

Simply change the model configuration in any existing demo:

```python
# Instead of:
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Use:
SAM2_CHECKPOINT = "./checkpoints/efficienttam_ti.pt"
SAM2_MODEL_CONFIG = "configs/efficienttam/efficienttam_ti.yaml"
```

That's it! The rest of your code remains exactly the same.

## 📊 Performance Comparison

| Model | Size | Speed | Memory Usage | Use Case |
|-------|------|-------|--------------|----------|
| SAM2 Large | ~856MB | Slower | High | Best quality |
| SAM2 Base+ | ~224MB | Medium | Medium | Balanced |
| EfficientTAM Small | ~130MB | Fast | Low | Efficient balanced |
| EfficientTAM Tiny | ~68MB | Fastest | Lowest | Maximum efficiency |

## 🔧 Available Models

### EfficientTAM Models

| Config File | Checkpoint | Description |
|-------------|------------|-------------|
| `configs/efficienttam/efficienttam_ti.yaml` | `efficienttam_ti.pt` | Tiny model - fastest inference |
| `configs/efficienttam/efficienttam_s.yaml` | `efficienttam_s.pt` | Small model - balanced performance |
| `configs/efficienttam/efficienttam_ti_512x512.yaml` | `efficienttam_ti_512x512.pt` | Tiny model optimized for 512x512 |
| `configs/efficienttam/efficienttam_s_512x512.yaml` | `efficienttam_s_512x512.pt` | Small model optimized for 512x512 |

## 💻 Usage Examples

### Example 1: Basic Image Segmentation

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load EfficientTAM model (same interface as SAM2!)
model = build_sam2(
    config_file="configs/efficienttam/efficienttam_ti.yaml",
    ckpt_path="checkpoints/efficienttam_ti.pt",
    device="cuda"
)

predictor = SAM2ImagePredictor(model)
predictor.set_image(your_image)

# Predict masks with bounding boxes
masks, scores, logits = predictor.predict(
    box=your_boxes,
    multimask_output=False
)
```

### Example 2: Video Tracking

```python
from sam2.build_sam import build_sam2_video_predictor

# Load EfficientTAM video predictor
video_predictor = build_sam2_video_predictor(
    config_file="configs/efficienttam/efficienttam_ti.yaml",
    ckpt_path="checkpoints/efficienttam_ti.pt"
)

# Use exactly like SAM2 video predictor
inference_state = video_predictor.init_state(video_path="your_video.mp4")
# ... rest of video tracking code remains the same
```

### Example 3: Modifying Existing Demos

For `grounded_sam2_dinox_demo.py`:

```python
# Change these lines at the top of the file:
SAM2_CHECKPOINT = "./checkpoints/efficienttam_ti.pt"
SAM2_MODEL_CONFIG = "configs/efficienttam/efficienttam_ti.yaml"

# Everything else remains exactly the same!
```

## 🧪 Testing the Integration

Run the provided test scripts to verify everything works:

```bash
# Test the integration
python test_efficienttam_integration.py

# Test with a demo-style example
python test_efficienttam_demo.py
```

## 🏗️ Architecture Overview

The integration works through several key components:

### 1. Auto-Detection System (`sam2/build_sam.py`)
- Automatically detects EfficientTAM configs based on file path
- Routes to appropriate builder functions transparently

### 2. EfficientTAM Adapter (`sam2/efficienttam_adapter.py`)
- Provides SAM2-compatible interface for EfficientTAM models
- Handles configuration and compilation settings
- Manages import paths and dependencies

### 3. Configuration Files (`sam2/configs/efficienttam/`)
- EfficientTAM model configurations copied for easy access
- Compatible with existing Hydra-based configuration system

## 🔍 Technical Details

### How It Works

1. **Config Detection**: When you call `build_sam2()` with an EfficientTAM config, the system automatically detects this and routes to EfficientTAM builders.

2. **Interface Compatibility**: EfficientTAM models expose the same methods as SAM2 models (`init_state`, `add_new_points_or_box`, `propagate`, etc.).

3. **Transparent Integration**: Your existing code doesn't need to change - just swap the config and checkpoint paths.

### Key Features

- ✅ **Drop-in Replacement**: Change only config/checkpoint paths
- ✅ **Same API**: All existing SAM2 methods work identically
- ✅ **Auto-Detection**: Automatically routes to correct backend
- ✅ **Performance**: Significantly faster and more memory-efficient
- ✅ **Compatibility**: Works with all existing demos and scripts

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure EfficientTAM is properly cloned in the right location:
   ```bash
   ls EfficientTAM/  # Should show the EfficientTAM repository
   ```

2. **Compilation Issues**: The integration automatically disables problematic torch compilation. If you still see issues, you can manually disable it:
   ```python
   model = build_sam2(
       config_file="configs/efficienttam/efficienttam_ti.yaml",
       ckpt_path="checkpoints/efficienttam_ti.pt",
       hydra_overrides_extra=["++model.compile_image_encoder=False"]
   )
   ```

3. **Memory Issues**: Use CPU for testing or smaller models:
   ```python
   model = build_sam2(..., device="cpu")
   ```

### Verification

Run the test suite to verify everything is working:

```bash
python test_efficienttam_integration.py
```

Expected output should show all tests passing with ✓ marks.

## 📚 References

- [EfficientTAM Repository](https://github.com/yformer/EfficientTAM)
- [Grounded-SAM-2 Repository](https://github.com/IDEA-Research/Grounded-SAM-2)
- [SAM2 Documentation](https://github.com/facebookresearch/sam2)

## 🤝 Contributing

If you encounter issues or have improvements:

1. Test with the provided test scripts
2. Check the troubleshooting section
3. Create an issue with detailed error messages and environment info

## 📄 License

This integration follows the same license terms as the original repositories:
- EfficientTAM: Apache 2.0 License
- Grounded-SAM-2: Apache 2.0 License
- SAM2: Apache 2.0 License 