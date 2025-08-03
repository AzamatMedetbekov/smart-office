from ultralytics import YOLO
import streamlit as st
import cv2 as cv
import settings
import logging

logger = logging.getLogger(__name__)


def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        st.error(f"Failed to load model: {e}")
        return None


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))
    is_display_tracker = display_tracker == "Yes"
    tracker_type = (
        st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        if is_display_tracker
        else None
    )
    return is_display_tracker, tracker_type


def _display_detected_frames(
    conf, model, st_frame, image, is_display_tracking=False, tracker=None
):
    try:
        if image is None or image.size == 0:
            return
            
        image = cv.resize(image, (720, int(720 * (9 / 16))))
        if is_display_tracking:
            res = model.track(image, conf=conf, persist=True, tracker=tracker)
        else:
            res = model.predict(image, conf=conf)
            
        if res and len(res) > 0:
            res_plotted = res[0].plot()
            st_frame.image(
                res_plotted, caption="Detected Video", channels="BGR", use_container_width=True
            )
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        st.error("Error processing frame")


def play_stored_video(conf, model):
    if model is None:
        st.error("Model not loaded")
        return
        
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    
    if st.sidebar.button("Detect Video Objects"):
        try:
            path = str(settings.VIDEOS_DICT[source_vid])
            vid_cap = cv.VideoCapture(path)
            
            if not vid_cap.isOpened():
                st.error("Failed to open video file")
                return
                
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(
                    conf, model, st_frame, image, is_display_tracker, tracker
                )
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            st.error(f"Error processing video: {e}")
        finally:
            if 'vid_cap' in locals():
                vid_cap.release()


def play_webcam(conf, model):
    if model is None:
        st.error("Model not loaded")
        return
        
    is_display_tracker, tracker = display_tracker_options()
    
    if st.sidebar.button("Detect Objects from Webcam"):
        try:
            vid_cap = cv.VideoCapture(settings.WEBCAM_PATH)
            
            if not vid_cap.isOpened():
                st.error("Cannot access webcam")
                return
                
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(
                    conf, model, st_frame, image, is_display_tracker, tracker
                )
        except Exception as e:
            logger.error(f"Webcam error: {e}")
            st.error(f"Webcam error: {e}")
        finally:
            if 'vid_cap' in locals():
                vid_cap.release()