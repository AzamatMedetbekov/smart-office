from pathlib import Path
import PIL
import streamlit as st
import settings
import helper

st.set_page_config(
    page_title="Smart Office Detection",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.markdown("""
    <style>
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #F8F9FA;
            padding: 1rem;
            border-right: 1px solid #eee;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Model Configuration")
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Model Confidence", 25, 100, 40)) / 100

model_path = settings.DETECTION_MODEL if model_type == 'Detection' else settings.SEGMENTATION_MODEL
model = helper.load_model(model_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎥 Input Source")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", 'bmp', 'webp'])
    col1, col2 = st.columns(2)
    with col1:
        if source_img:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image", use_container_width=True)
        else:
            default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
            st.image(default_image, caption="Default Image", use_container_width=True)
    with col2:
        if st.sidebar.button("🚀 Detect Objects"):
            img = uploaded_image if source_img else default_image
            res = model.predict(img, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_container_width=True)
            with st.expander("🔍 Detection Results"):
                for box in boxes:
                    st.write(box.data)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")