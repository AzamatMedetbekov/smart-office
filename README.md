# Smart Office Detection: YOLOv11 + SAHI + Streamlit

This project showcases a full **Smart Office object detection and segmentation** pipeline powered by **YOLOv11** and **Streamlit**, enhanced with **SAHI (Slicing Aided Hyper Inference)** for detecting small and overlapping objects. It provides a clean, interactive web UI for image inference tasks.

---

## Demo (Streamlit WebApp)

> ⚠️ Streamlit Cloud demo link coming soon. Locally, run the app using the instructions below.

---

## Features

* **Object Detection** using YOLOv11 (custom fine-tuned weights)
* **Instance Segmentation** with YOLOv11L-seg model
* **SAHI** slicing support for better detection of small/overlapping objects
* Upload and detect objects in images (videos and streams coming soon)
* Clean, responsive UI with Streamlit

---

## Project Structure

```
smart-office/
├── app.py                  # Streamlit app
├── helper.py              # Helper methods for webcam/video/image inference
├── sahi_helper.py         # SAHI integration
├── settings.py            # Path config
├── weights/
│   ├── yolo11l.pt         # Detection model
│   └── yolov11l-seg.pt    # Segmentation model
├── images/
│   └── office_4.jpg       # Default image
├── videos/                # Video samples
├── requirements.txt       # Dependencies
└── cv-project-training.ipynb # Training notebook
```

---

## Installation

> Python 3.10+ is recommended

```bash
# Clone this repository
git clone https://github.com/your-username/smart-office-detection.git
cd smart-office-detection

# Install dependencies
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit opencv-python numpy matplotlib Pillow \
            scikit-image tqdm PyYAML requests scipy \
            sahi ultralytics torch torchvision \
            cvzone filterpy lap hydra-core yt-dlp
```

---

## Model Weights

Place your model weights in the `weights/` directory:

* **Detection**: `weights/yolo11l.pt`
* **Segmentation**: `weights/yolov11l-seg.pt`

If you don’t have weights, train using the steps below or download YOLOv11 models from Ultralytics.

---

## Run the App

```bash
streamlit run app.py
```

---

## Inference Workflow

1. Select Task: Detection or Segmentation
2. Set Confidence threshold (slider)
3. Toggle SAHI for slicing inference
4. Upload Image or use default sample
5. Click 'Detect Objects' to see results

*Only image input is supported in this version.*

---

## Training a Custom Model

We used the dataset [Office Object Detection Dataset](https://www.kaggle.com/datasets/walidguirat/office-object-detection) from Kaggle.

Training was done using YOLOv11L model on this dataset. Refer to the provided notebook `cv-project-training.ipynb` for full training configuration.

Example training:

```python
from ultralytics import YOLO
model = YOLO("yolo11l.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    cache=True,
    amp=True,
    project="runs/office",
    name="yolov11l_office"
)
```

Minimal `data.yaml`:

```yaml
train: data/images/train
val: data/images/val

nc: 5
names: [chair, dining table, keyboard, laptop, person]
```

Make sure label files (`.txt`) have correct class indices (0 to 4).

---

## SAHI Inference (Post-training)

```python
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/office/yolov11l_office/weights/best.pt",
    confidence_threshold=0.3,
)

result = get_sliced_prediction(
    image="your_image.jpg",
    detection_model=model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

*SAHI is used after model training to improve detection of small/overlapping objects.*

---

## Screenshots

| Upload | Detection | SAHI |
| ------ | --------- | ---- |
|        |           |      |

---

## Evaluation Script

Evaluation will be added after training completes. It will be used to assess mAP and accuracy metrics.

---

## requirements.txt

See the provided file. Main packages include:

* `ultralytics>=8.2.0,<12.0.0`
* `sahi>=0.11.14`
* `torch`, `torchvision`
* `streamlit`, `opencv-python`, `matplotlib`, etc.

---

## Tech Stack

* **YOLOv11** by Ultralytics for object detection/segmentation
* **Streamlit** for front-end visualization
* **SAHI** for improved slicing-based inference
* **Kaggle Notebooks** for training

---

## Notes

* Kaggle Notebooks will stop training when tab/browser closes unless GPU is persistent
* SAHI helps when objects are small, dense, or overlapping
* Segmentation model used: `yolov11l-seg.pt`

---

If you found this helpful, please ⭐️ the repo and share!
