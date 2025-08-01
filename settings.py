from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent

if ROOT not in sys.path:
    sys.path.append(str(ROOT))

ROOT = ROOT.relative_to(Path.cwd())

# Supported input sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Paths
IMAGES_DIR = ROOT / 'images'
VIDEO_DIR = ROOT / 'videos'
MODEL_DIR = ROOT / 'weights'

DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'video_1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
}

DETECTION_MODEL = MODEL_DIR / 'Yolo-Weights/yolo11l.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov11n-seg.pt'

WEBCAM_PATH = 0
