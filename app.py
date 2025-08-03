from pathlib import Path
import PIL
import streamlit as st
import settings
import helper
from sahi_helper import run_sahi_inference
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Smart Office Detection",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #F8F9FA;
            padding: 1.5rem;
            border-right: 1px solid #e0e0e0;
        }
        h1, h2, h3 {
            color: #7950F2;
        }
        .css-1v0mbdj p {
            color: #1E1E1E;
        }
        .stButton > button {
            background-color: #7950F2;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #6741d9;
        }
        .stRadio > div {
            color: #1E1E1E;
        }
        .stSelectbox > div > div {
            color: #1E1E1E;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1239/1239710.png", width=100)
st.sidebar.markdown("### ⚙️ **Model Configuration**")
model_type = st.sidebar.radio("Select Task", ["Detection", "Segmentation"])
confidence = float(st.sidebar.slider("Model Confidence", 25, 100, 40)) / 100
use_sahi = st.sidebar.checkbox("🧩 Use SAHI Slicing", value=False)

try:
    model_path = (
        settings.DETECTION_MODEL
        if model_type == "Detection"
        else settings.SEGMENTATION_MODEL
    )
    model = helper.load_model(model_path)
    if model is None:
        st.error("❌ Failed to load model. Please check model path.")
        st.stop()
except Exception as e:
    logger.error(f"Model loading error: {e}")
    st.error(f"❌ Error loading model: {e}")
    st.stop()

try:
    class_names = model.names if hasattr(model, "names") else model.model.names
    if isinstance(class_names, dict):
        all_classes = list(class_names.values())
    else:
        all_classes = list(class_names)

    DESIRED = [
        "cell phone",
        "mouse",
        "bottle",
        "cup",
        "backpack",
        "tv",
        "person",
        "chair",
    ]

    available = [c for c in all_classes if c in DESIRED]

    selected_classes = st.sidebar.multiselect(
        "🔍 Filter Classes",
        options=available,
        default=available,
        help="Only show these categories in the result list",
    )
except Exception as e:
    logger.warning(f"Could not extract class names: {e}")
    selected_classes = []

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎥 **Input Source**")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

st.title("📸 Smart Office Image Detection")
st.markdown(
    "Use your custom YOLOv11 or SAHI-enhanced object detector on uploaded images."
)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("#### 🖼️ Original Image")
        try:
            if source_img:
                uploaded_image = PIL.Image.open(source_img).convert("RGB")
                image_np = np.array(uploaded_image)
                st.image(
                    uploaded_image, caption="Uploaded Image", use_container_width=True
                )
            else:
                if (
                    hasattr(settings, "DEFAULT_IMAGE")
                    and Path(settings.DEFAULT_IMAGE).exists()
                ):
                    default_image = PIL.Image.open(settings.DEFAULT_IMAGE).convert(
                        "RGB"
                    )
                    image_np = np.array(default_image)
                    st.image(
                        default_image, caption="Default Image", use_container_width=True
                    )
                else:
                    st.info("📤 Please upload an image to get started.")
                    image_np = None
                    default_image = None
        except Exception as e:
            logger.error(f"Image loading error: {e}")
            st.error(f"❌ Error loading image: {e}")
            image_np = None
            default_image = None

    with col2:
        st.markdown("#### 🧠 Detection Result")
        if st.sidebar.button("🚀 Detect Objects"):
            if image_np is None:
                st.warning("⚠️ Please upload a valid image first.")
            else:
                try:
                    with st.spinner("🔄 Processing image..."):
                        if use_sahi:
                            sahi_img, result = run_sahi_inference(
                                image_np,
                                str(model_path),
                                conf=confidence,
                                filter_classes=(
                                    selected_classes if selected_classes else None
                                ),
                            )

                            if sahi_img is not None and result is not None:
                                st.image(
                                    sahi_img,
                                    caption="SAHI Sliced Detection",
                                    channels="RGB",
                                    use_container_width=True,
                                )
                                with st.expander("🔍 Detection Results"):
                                    if result.object_prediction_list:
                                        filtered_count = 0
                                        for pred in result.object_prediction_list:
                                            if (
                                                not selected_classes
                                                or pred.category.name
                                                in selected_classes
                                            ):
                                                filtered_count += 1
                                                st.write(
                                                    {
                                                        "label": pred.category.name,
                                                        "confidence": round(
                                                            pred.score.value, 3
                                                        ),
                                                        "bbox": (
                                                            pred.bbox.to_xywh()
                                                            if hasattr(
                                                                pred.bbox, "to_xywh"
                                                            )
                                                            else str(pred.bbox)
                                                        ),
                                                    }
                                                )

                                        if filtered_count == 0:
                                            st.info(
                                                "No objects detected matching selected classes."
                                            )
                                        else:
                                            st.info(
                                                f"Showing {filtered_count} detections"
                                            )
                                    else:
                                        st.info("No objects detected.")
                            else:
                                st.error("❌ SAHI processing failed. Please try again.")

                        else:
                            img = uploaded_image if source_img else default_image
                            if img is None:
                                st.error("❌ No image available for processing.")

                            res = model.predict(img, conf=confidence)

                            if res and len(res) > 0:
                                boxes = res[0].boxes

                                if boxes is not None and len(boxes) > 0:
                                    frame = cv2.cvtColor(
                                        np.array(img), cv2.COLOR_RGB2BGR
                                    )

                                    filtered_count = 0
                                    detection_results = []

                                    for b in boxes:
                                        try:
                                            cid = int(b.cls.cpu().item())
                                            class_name = (
                                                all_classes[cid]
                                                if cid < len(all_classes)
                                                else f"class_{cid}"
                                            )

                                            if (
                                                not selected_classes
                                                or class_name in selected_classes
                                            ):
                                                filtered_count += 1
                                                x1, y1, x2, y2 = map(
                                                    int, b.xyxy[0].cpu().numpy()
                                                )
                                                conf_score = b.conf.cpu().item()

                                                cv2.rectangle(
                                                    frame,
                                                    (x1, y1),
                                                    (x2, y2),
                                                    (0, 255, 0),
                                                    2,
                                                )
                                                cv2.putText(
                                                    frame,
                                                    f"{class_name} {conf_score:.2f}",
                                                    (x1, y1 - 5),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (0, 255, 0),
                                                    1,
                                                )

                                                detection_results.append(
                                                    {
                                                        "label": class_name,
                                                        "confidence": round(
                                                            conf_score, 3
                                                        ),
                                                        "bbox": [
                                                            round(x1, 1),
                                                            round(y1, 1),
                                                            round(x2, 1),
                                                            round(y2, 1),
                                                        ],
                                                    }
                                                )
                                        except Exception as e:
                                            logger.error(f"Error processing box: {e}")
                                            continue

                                    st.image(
                                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                        caption="Filtered YOLO Detection",
                                        use_container_width=True,
                                    )

                                    with st.expander("🔍 Detection Results"):
                                        if detection_results:
                                            for result in detection_results:
                                                st.write(result)
                                            st.info(
                                                f"Showing {len(detection_results)} detections"
                                            )
                                        else:
                                            st.info(
                                                "No objects detected matching selected classes."
                                            )
                                else:
                                    st.info("No objects detected.")
                            else:
                                st.warning("⚠️ No detection results returned.")

                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    st.error(f"❌ Detection failed: {e}")

else:
    st.warning("🚧 Currently, only image detection is supported in this demo UI.")
    st.info("💡 Video and webcam support will be added in future updates.")
