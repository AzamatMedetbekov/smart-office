from pathlib import Path
import PIL
import streamlit as st
import settings
import helper

st.set_page_config(
    page_title="Smart Office - YOLOv11",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Smart Office: Object Detection using YOLOv11")

st.sidebar.header("Model Configuration")
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Model Confidence", 25, 100, 40)) / 100

model_path = Path(settings.DETECTION_MODEL if model_type == 'Detection' else settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model from path: {model_path}")
    st.error(ex)

st.sidebar.header("Source Selection")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_container_width=True)
            else:
                default_image = PIL.Image.open(str(settings.DEFAULT_IMAGE))
                st.image(default_image, caption="Default Image", use_container_width=True)
        except Exception as ex:
            st.error("Error opening image")
            st.error(ex)

    with col2:
        if st.sidebar.button('Detect Objects'):
            img = uploaded_image if source_img else default_image
            res = model.predict(img, conf=confidence)
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detection Result', use_container_width=True)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type.")