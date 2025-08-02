from pathlib import Path
import PIL
import streamlit as st
import settings
import helper
from sahi_helper import run_sahi_inference

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
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-right: 1px solid #d1d1d1;
        }
        h1, h2, h3 {
            color: #00416A;
        }
        .css-1v0mbdj p {
            color: #333333;
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

model_path = (
    settings.DETECTION_MODEL
    if model_type == "Detection"
    else settings.SEGMENTATION_MODEL
)
model = helper.load_model(model_path)

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
        if source_img:
            uploaded_image = PIL.Image.open(source_img)
            uploaded_image.convert("RGB").save("temp.jpg")
            st.image(source_img, caption="Uploaded Image", use_container_width=True)
        else:
            default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
            default_image.convert("RGB").save("temp.jpg")
            st.image(default_image, caption="Default Image", use_container_width=True)

    with col2:
        st.markdown("#### 🧠 Detection Result")
        if st.sidebar.button("🚀 Detect Objects"):
            if use_sahi:
                sahi_img, result = run_sahi_inference(
                    "temp.jpg", str(model_path), conf=confidence
                )
                st.image(
                    sahi_img,
                    caption="SAHI Sliced Detection",
                    channels="RGB",
                    use_container_width=True,
                )
                with st.expander("🔍 Detection Results"):
                    for pred in result.object_prediction_list:
                        st.write(
                            {
                                "label": pred.category.name,
                                "confidence": round(pred.score.value, 3),
                                "bbox": pred.bbox.to_xywh(),
                            }
                        )
            else:
                img = uploaded_image if source_img else default_image
                res = model.predict(img, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(
                    res_plotted,
                    caption="YOLOv11 Detection",
                    use_container_width=True,
                )
                with st.expander("🔍 Detection Results"):
                    for box in boxes:
                        st.write(box.data)

else:
    st.warning("🚧 Currently, only image detection is supported in this demo UI.")
