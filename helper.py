from ultralytics import YOLO
import streamlit as st
import cv2 as cv
import settings

def load_model(model_path):
    return YOLO(model_path)

def _display_detected_frames(conf, model, st_frame, image):
    image = cv.resize(image, (720, int(720 * (9/16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Frame', channels="BGR", use_container_width=True)

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    if st.sidebar.button('Detect Video Objects'):
        try:
            path = str(settings.VIDEOS_DICT[source_vid])
            vid_cap = cv.VideoCapture(path)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(conf, model, st_frame, image)
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"Error loading video: {e}")

def play_webcam(conf, model):
    if st.sidebar.button("Start Webcam Detection"):
        try:
            vid_cap = cv.VideoCapture(settings.WEBCAM_PATH)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(conf, model, st_frame, image)
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"Webcam error: {e}")
