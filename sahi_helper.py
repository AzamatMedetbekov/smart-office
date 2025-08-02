from sahi.predict import get_sliced_prediction
from sahi.auto_model import AutoDetectionModel
import cv2
import numpy as np
import torch


def draw_boxes(image: np.ndarray, object_predictions, box_color=(0, 255, 0)) -> np.ndarray:
    for prediction in object_predictions:
        bbox = prediction.bbox
        x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
        label = prediction.category.name
        conf = prediction.score.value
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
    return image


def run_sahi_inference(
    img_path,
    model_path,
    conf=0.3,
    slice_size=256,
    overlap=0.2,
):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        image_size=640,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    image = cv2.imread(img_path)
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )

    image_with_boxes = draw_boxes(image.copy(), result.object_prediction_list)
    image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

    return image_rgb, result
