from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

IMAGE = "images"
VIDEO = "Video"
WEBCAM = "Webcam"
SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

IMAGES_DIR = ROOT / "images"
VIDEO_DIR = ROOT / "videos"
MODEL_DIR = ROOT / "weights"

DEFAULT_IMAGE = IMAGES_DIR / "temp.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "temp.jpg"

VIDEOS_DICT = {
    "Office Footage 1": VIDEO_DIR / "video_1.mp4",
    "Office Footage 2": VIDEO_DIR / "video_2.mp4",
    "Office Footage 3": VIDEO_DIR / "video_3.mp4",
}

DETECTION_MODEL = MODEL_DIR / "yolo11l.pt"
SEGMENTATION_MODEL = MODEL_DIR / "yolo11l-seg.pt"

WEBCAM_PATH = 0  