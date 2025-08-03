from sahi.predict import get_sliced_prediction
from sahi.auto_model import AutoDetectionModel
import cv2
import numpy as np
import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

_model_cache = {}

def draw_boxes(image: np.ndarray, object_predictions, box_color=(0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes on image with error handling."""
    try:
        for prediction in object_predictions:
            bbox = prediction.bbox
            x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            label = prediction.category.name
            conf = prediction.score.value
            
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                image,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
            )
    except Exception as e:
        logger.error(f"Error drawing boxes: {e}")
    
    return image

def get_or_load_model(model_path: str, conf: float = 0.3) -> Optional[AutoDetectionModel]:
    """Load model with caching and error handling."""
    cache_key = f"{model_path}_{conf}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {device}")
        
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            image_size=640,
            device=device,
        )
        
        _model_cache[cache_key] = detection_model
        return detection_model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None

def run_sahi_inference(
    image: np.ndarray,
    model_path: str,
    conf: float = 0.3,
    slice_size: int = 256,
    overlap: float = 0.2,
) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """Run SAHI inference with comprehensive error handling."""
    
    if image is None or image.size == 0:
        logger.error("Invalid input image")
        return None, None
    
    try:
        detection_model = get_or_load_model(model_path, conf)
        if detection_model is None:
            return None, None

        try:
            result = get_sliced_prediction(
                image=image,
                detection_model=detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM, clearing cache and retrying on CPU")
                torch.cuda.empty_cache()
                detection_model.model.to("cpu")
                result = get_sliced_prediction(
                    image=image,
                    detection_model=detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap,
                    overlap_width_ratio=overlap,
                )
            else:
                raise

        image_with_boxes = draw_boxes(image.copy(), result.object_prediction_list)
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

        return image_rgb, result

    except Exception as e:
        logger.error(f"SAHI inference failed: {e}")
        return None, None

def cleanup_models():
    """Clean up cached models and free GPU memory."""
    global _model_cache
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")