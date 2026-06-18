import os
import av
import cv2
import time
import urllib.request
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Hand Tracking",
    page_icon="✋",
    layout="wide",
)

# Config of the ICE (STUN/TURN) servers
# -----------------------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="hand-tracking",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,  # Ajout de cette ligne
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Model download (the modern MediaPipe Tasks API needs a .task model file;
# it is not bundled in the pip package, so we fetch it once and cache it
# on disk. This replaces the legacy `mp.solutions.hands` API, which has
# been removed in current MediaPipe releases.)
# -----------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@st.cache_resource
def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading hand-tracking model (first run only)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

# -----------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------
st.sidebar.title("✋ Hand Tracking")
st.sidebar.markdown(
    "Real-time hand landmark detection running **in your browser**, "
    "powered by [MediaPipe](https://developers.google.com/mediapipe) and "
    "[OpenCV](https://opencv.org/), streamed live via WebRTC."
)

st.sidebar.divider()
st.sidebar.subheader("Settings")

max_hands = st.sidebar.slider(
    "Max hands to detect", 1, 4, 2,
    help="The maximum number of hands the model will track at once. "
         "Set to 1 if only one hand will be in frame — it's slightly faster "
         "and avoids occasional false detections.",
)
detection_confidence = st.sidebar.slider(
    "Detection confidence", 0.1, 1.0, 0.6, 0.05,
    help="How confident the model must be before it agrees a hand is present "
         "at all. Lower this if your hand isn't being detected (e.g. in dim "
         "lighting); raise it if it's detecting hands that aren't really there.",
)
tracking_confidence = st.sidebar.slider(
    "Tracking confidence", 0.1, 1.0, 0.6, 0.05,
    help="How confident the model must be to keep following an already-"
         "detected hand between frames. Lower this if tracking keeps dropping "
         "out during fast movement; raise it for steadier, more conservative tracking.",
)
show_fps = st.sidebar.checkbox(
    "Show FPS counter", value=True,
    help="Displays how many frames per second are being processed, "
         "in the top-left corner of the video.",
)
show_finger_count = st.sidebar.checkbox(
    "Show finger count", value=True,
    help="Displays how many fingers are raised on each detected hand, "
         "labeled with which hand (Left/Right) it belongs to.",
)
draw_style = st.sidebar.radio(
    "Landmark style", ["Skeleton", "Dots only"], index=0,
    help="'Skeleton' draws the 21 hand points connected by lines, showing the "
         "hand's structure. 'Dots only' shows just the points, with no lines.",
)

st.sidebar.divider()
st.sidebar.caption(
    "Tip: allow camera access when your browser prompts you. "
    "Nothing is recorded or sent anywhere — frames are processed live and discarded."
)

# -----------------------------------------------------------------------
# MediaPipe setup (Tasks API)
# -----------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

SIGNAL_COLOR = (212, 234, 94)   # BGR for #5EEAD4
AMBER_COLOR = (90, 166, 242)    # BGR for #F2A65A


def count_raised_fingers(landmarks):
    """Counts extended fingers from 21 hand landmarks.

    Four fingers (index/middle/ring/pinky): a finger is 'up' if its tip sits
    higher on screen (smaller y) than its own pip joint.

    Thumb: rather than comparing x-position (which depends on knowing the
    hand's left/right side and gets confused by mirroring), we compare the
    thumb tip's distance from the pinky-side of the palm (landmark 17)
    against the thumb's own base joint's distance from that same point. An
    extended thumb sticks out away from the palm, so its tip ends up farther
    from landmark 17 than its base joint is. This works the same regardless
    of which hand it is or whether the image is mirrored.
    """
    def dist(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    fingers_up = 0

    if dist(landmarks[4], landmarks[17]) > dist(landmarks[2], landmarks[17]):
        fingers_up += 1

    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers_up += 1

    return fingers_up


class HandTrackingProcessor:
    """Wraps a MediaPipe Tasks HandLandmarker for per-frame video processing.

    The landmarker itself is created once per WebRTC session (recreating it
    every frame would be too slow), but its tunable parameters are re-applied
    whenever the sidebar values change, by recreating the landmarker lazily
    inside recv() if the settings differ from what it was built with.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.landmarker = None
        self._built_with = None  # (max_hands, det_conf, track_conf) the landmarker was built with
        self._prev_time = time.time()
        self._start_time = time.time()
        # Live-updatable settings, set from outside via ctx.video_processor
        self.max_hands = 2
        self.detection_confidence = 0.6
        self.tracking_confidence = 0.6
        self.show_fps = True
        self.show_finger_count = True
        self.draw_style = "Skeleton"

    def _ensure_landmarker(self):
        wanted = (self.max_hands, self.detection_confidence, self.tracking_confidence)
        if self.landmarker is None or self._built_with != wanted:
            if self.landmarker is not None:
                self.landmarker.close()
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
            self.landmarker = HandLandmarker.create_from_options(options)
            self._built_with = wanted

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._ensure_landmarker()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror for a natural "looking in a mirror" feel
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            handedness_label = "Right"
            if idx < len(result.handedness) and result.handedness[idx]:
                handedness_label = result.handedness[idx][0].category_name or "Right"

            # MediaPipe classifies handedness from the frame it actually sees.
            # We feed it the mirrored (flipped) frame so the live view looks
            # natural, which means its Left/Right label is the mirror image
            # of what the viewer perceives as their own hand. Swap it back.
            handedness_label = "Left" if handedness_label == "Right" else "Right"

            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            if self.draw_style == "Skeleton":
                for conn in HAND_CONNECTIONS:
                    cv2.line(img, points[conn.start], points[conn.end], SIGNAL_COLOR, 2)
                for (cx, cy) in points:
                    cv2.circle(img, (cx, cy), 4, SIGNAL_COLOR, cv2.FILLED)
            else:
                for (cx, cy) in points:
                    cv2.circle(img, (cx, cy), 5, SIGNAL_COLOR, cv2.FILLED)

            if self.show_finger_count:
                count = count_raised_fingers(hand_landmarks)
                wrist_x, wrist_y = points[0]
                label_pos = (max(wrist_x - 30, 10), max(wrist_y + 40, 30))
                cv2.putText(
                    img, f"{handedness_label}: {count}", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER_COLOR, 2, cv2.LINE_AA,
                )

        if self.show_fps:
            now = time.time()
            fps = 1 / (now - self._prev_time) if now != self._prev_time else 0
            self._prev_time = now
            cv2.putText(
                img, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, SIGNAL_COLOR, 2, cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------
st.title("✋ Real-Time Hand Tracking")

model_path = ensure_model_downloaded()

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def processor_factory():
    return HandTrackingProcessor(model_path)


ctx = webrtc_streamer(
    key="hand-tracking",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=processor_factory,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Push the current sidebar values into the running processor (if active)
# so changes take effect live without restarting the stream.
if ctx.video_processor:
    ctx.video_processor.max_hands = max_hands
    ctx.video_processor.detection_confidence = detection_confidence
    ctx.video_processor.tracking_confidence = tracking_confidence
    ctx.video_processor.show_fps = show_fps
    ctx.video_processor.show_finger_count = show_finger_count
    ctx.video_processor.draw_style = draw_style

st.divider()

st.markdown(
    "An improved, browser-deployable version of a classic OpenCV + MediaPipe "
    "hand-tracking script — rebuilt with **streamlit-webrtc** so it runs live "
    "against *your* webcam directly from this page, with adjustable detection "
    "settings and live finger counting."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("How it works")
    st.markdown(
        "- MediaPipe detects **21 landmarks** per hand\n"
        "- Each frame is processed **live in your browser session**\n"
        "- Finger count uses landmark geometry, no extra model\n"
        "- Nothing is stored or uploaded"
    )

with col2:
    st.subheader("Built with")
    st.markdown("`OpenCV` · `MediaPipe` · `Streamlit` · `streamlit-webrtc`")

st.caption(
    "Original concept inspired by open-source OpenCV/MediaPipe hand-tracking scripts, "
    "rebuilt from the ground up for live browser deployment."
)