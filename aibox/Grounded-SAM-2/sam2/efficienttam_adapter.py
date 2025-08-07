"""
Adapter that lets Grounded-SAM 2 load EfficientTAM via the usual build_sam2_* entry points.

This adapter provides a transparent interface so that existing demo code can use
EfficientTAM models by simply changing the config file and checkpoint path.
"""

import sys
import os
from pathlib import Path
import torch
from collections import OrderedDict

# Add EfficientTAM to Python path
current_dir = Path(__file__).parent
efficienttam_path = current_dir.parent / "EfficientTAM"
if efficienttam_path.exists():
    sys.path.insert(0, str(efficienttam_path))

try:
    from efficient_track_anything.build_efficienttam import (
        build_efficienttam_video_predictor,
        build_efficienttam,
    )
    from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor
    
    EFFICIENTTAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EfficientTAM not available: {e}")
    EFFICIENTTAM_AVAILABLE = False


def build_sam2(config_file, ckpt_path=None, **kwargs):
    """
    Build EfficientTAM model with the same interface as build_sam2.
    
    Args:
        config_file: Path to EfficientTAM config file
        ckpt_path: Path to EfficientTAM checkpoint
        **kwargs: Additional arguments passed to build_efficienttam
    
    Returns:
        EfficientTAM model instance
    """
    if not EFFICIENTTAM_AVAILABLE:
        raise ImportError("EfficientTAM is not available. Please ensure it's properly installed.")
    
    # Disable compilation for streaming compatibility
    hydra_overrides_extra = kwargs.get('hydra_overrides_extra', [])
    if not any('compile_image_encoder' in override for override in hydra_overrides_extra):
        hydra_overrides_extra = hydra_overrides_extra + ["++model.compile_image_encoder=False"]
        kwargs['hydra_overrides_extra'] = hydra_overrides_extra
    
    return build_efficienttam(config_file, ckpt_path, **kwargs)


def build_sam2_video_predictor(config_file, ckpt_path=None, **kwargs):
    """
    Build EfficientTAM video predictor with the same interface as build_sam2_video_predictor.
    
    Args:
        config_file: Path to EfficientTAM config file
        ckpt_path: Path to EfficientTAM checkpoint
        **kwargs: Additional arguments passed to build_efficienttam_video_predictor
    
    Returns:
        EfficientTAM video predictor instance
    """
    if not EFFICIENTTAM_AVAILABLE:
        raise ImportError("EfficientTAM is not available. Please ensure it's properly installed.")
    
    # Disable compilation for streaming compatibility
    hydra_overrides_extra = kwargs.get('hydra_overrides_extra', [])
    if not any('compile_image_encoder' in override for override in hydra_overrides_extra):
        hydra_overrides_extra = hydra_overrides_extra + ["++model.compile_image_encoder=False"]
        kwargs['hydra_overrides_extra'] = hydra_overrides_extra
    
    return build_efficienttam_video_predictor(config_file, ckpt_path, **kwargs)


# Alias for image predictor compatibility
SAM2ImagePredictor = EfficientTAMImagePredictor if EFFICIENTTAM_AVAILABLE else None


def is_efficienttam_config(config_file):
    """
    Check if a config file is for EfficientTAM based on the path.
    
    Args:
        config_file: Path to config file
        
    Returns:
        bool: True if this appears to be an EfficientTAM config
    """
    config_path = str(config_file).lower()
    return 'efficienttam' in config_path


def get_efficienttam_models():
    """
    Get list of available EfficientTAM model configurations.
    
    Returns:
        dict: Mapping of model names to (config_file, checkpoint_name) tuples
    """
    if not EFFICIENTTAM_AVAILABLE:
        return {}
    
    models = {
        "efficienttam_ti": ("EfficientTAM/efficient_track_anything/configs/efficienttam/efficienttam_ti.yaml", "efficienttam_ti.pt"),
        "efficienttam_ti_512": ("EfficientTAM/efficient_track_anything/configs/efficienttam/efficienttam_ti_512x512.yaml", "efficienttam_ti_512x512.pt"),
        "efficienttam_s": ("EfficientTAM/efficient_track_anything/configs/efficienttam/efficienttam_s.yaml", "efficienttam_s.pt"),
        "efficienttam_s_512": ("EfficientTAM/efficient_track_anything/configs/efficienttam/efficienttam_s_512x512.yaml", "efficienttam_s_512x512.pt"),
    }
    return models 


class EfficientTAMAdapter:
    """
    Adapter to make EfficientTAM compatible with SAM2 API.
    
    This adapter wraps EfficientTAM models to provide the same interface as SAM2,
    allowing drop-in replacement while maintaining full compatibility.
    """
    
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = device
        
        # Build EfficientTAM model
        self.model = build_efficienttam_video_predictor(
            config_path, 
            checkpoint_path, 
            device=device
        )
        
        # Forward essential attributes
        self.image_size = self.model.image_size
        
    def forward_image(self, img_batch):
        """Forward image through the model - compatible with SAM2 API"""
        return self.model.forward_image(img_batch)
    
    @torch.inference_mode()
    def init_state(
        self,
        video_path=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """
        Initialize an inference state - SAM2 compatible version.
        
        This method handles video_path=None properly for streaming mode,
        unlike the original EfficientTAM implementation.
        """
        compute_device = self.device
        inference_state = {}
        
        if video_path is not None:
            # Use original EfficientTAM init_state for video files
            return self.model.init_state(
                video_path=video_path,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                async_loading_frames=async_loading_frames,
            )
        else:
            # **FIX**: Handle streaming mode (video_path=None) like SAM2
            print("Real-time streaming mode: waiting for first image input...")
            images = None
            video_height, video_width = None, None
            inference_state["images"] = None
            inference_state["num_frames"] = 0

        # Set up inference state structure (matching EfficientTAM format)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
            
        # Initialize object tracking structures
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        
        # EfficientTAM-specific structures
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        
        # Don't warm up backbone for streaming mode (no frame 0 yet)
        return inference_state
    
    def add_new_frame(self, inference_state, new_image):
        """Add new frame - delegate to EfficientTAM"""
        return self.model.add_new_frame(inference_state, new_image)
    
    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points=None, labels=None, box=None, **kwargs):
        """Add new points or box - delegate to EfficientTAM"""
        return self.model.add_new_points_or_box(
            inference_state, frame_idx, obj_id, 
            points=points, labels=labels, box=box, **kwargs
        )
    
    def infer_single_frame(self, inference_state, frame_idx):
        """Infer single frame - delegate to EfficientTAM"""
        return self.model.infer_single_frame(inference_state, frame_idx)
    
    def reset_state(self, inference_state):
        """Reset state - delegate to EfficientTAM"""
        return self.model.reset_state(inference_state)
    
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """Remove object - delegate to EfficientTAM"""
        return self.model.remove_object(inference_state, obj_id, strict=strict, need_output=need_output)
    
    def __getattr__(self, name):
        """Forward any other attributes to the underlying EfficientTAM model"""
        return getattr(self.model, name) 