from ultralytics import YOLO
import streamlit as st
import cv2 as cv
import settings

def load_model(model_path):
    return YOLO(model_path)

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = display_tracker == 'Yes'
    tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml")) if is_display_tracker else None
    return is_display_tracker, tracker_type

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=False, tracker=None):
    image = cv.resize(image, (720, int(720 * (9/16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_container_width=True)

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Video Objects'):
        path = str(settings.VIDEOS_DICT[source_vid])
        vid_cap = cv.VideoCapture(path)
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if not success:
                break
            _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
        vid_cap.release()

def play_webcam(conf, model):
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects from Webcam'):
        vid_cap = cv.VideoCapture(settings.WEBCAM_PATH)
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if not success:
                break
            _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
        vid_cap.release()

