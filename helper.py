# --- helper.py ---
from ultralytics import YOLO
import streamlit as st
import cv2 as cv
import yt_dlp
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

def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]', 'no_warnings': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def play_video(conf, model, source, label, example_url=None):
    stream_url = st.sidebar.text_input(label)
    if example_url:
        st.sidebar.caption(f'Example URL: {example_url}')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv.VideoCapture(stream_url)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            stream_url = get_youtube_stream_url(source_youtube)
            vid_cap = cv.VideoCapture(stream_url)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Video Objects'):
        try:
            path = str(settings.VIDEOS_DICT[source_vid])
            vid_cap = cv.VideoCapture(path)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"Error loading video: {e}")

def play_webcam(conf, model):
    play_video(conf, model, settings.WEBCAM_PATH, "Webcam Stream Path")

def play_rtsp_stream(conf, model):
    play_video(conf, model, None, "RTSP Stream URL", 'rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')