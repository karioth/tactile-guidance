"""
Smart Image Predictor that automatically chooses between SAM2 and EfficientTAM
based on the model type.
"""

from .sam2_image_predictor import SAM2ImagePredictor


def SmartImagePredictor(sam_model):
    """
    Factory function that returns the appropriate image predictor based on the model type.
    
    Args:
        sam_model: The SAM model instance (either SAM2 or EfficientTAM)
        
    Returns:
        Appropriate image predictor instance
    """
    # Check if this is an EfficientTAM model
    model_class_name = sam_model.__class__.__name__
    
    if 'EfficientTAM' in model_class_name:
        try:
            from .efficienttam_adapter import SAM2ImagePredictor as EfficientTAMImagePredictor
            if EfficientTAMImagePredictor is not None:
                print("Using EfficientTAM Image Predictor")
                return EfficientTAMImagePredictor(sam_model)
        except ImportError:
            pass
    
    # Default to SAM2ImagePredictor
    print("Using SAM2 Image Predictor")
    return SAM2ImagePredictor(sam_model) 