"""Streamlit page layout: sidebar controls and static page content."""
import streamlit as st

from .config import Settings


def render_sidebar() -> Settings:
    st.sidebar.title("✋ Hand Tracking")
    st.sidebar.markdown(
        "Real-time hand landmark detection on **your own webcam**, "
        "powered by [MediaPipe](https://developers.google.com/mediapipe) and "
        "[OpenCV](https://opencv.org/), streamed live via WebRTC."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Settings")

    settings = Settings(
        max_hands=st.sidebar.slider(
            "Max hands to detect", 1, 4, 2,
            help="The maximum number of hands the model will track at once. "
                 "Set to 1 if only one hand will be in frame — it's slightly faster "
                 "and avoids occasional false detections.",
        ),
        detection_confidence=st.sidebar.slider(
            "Detection confidence", 0.1, 1.0, 0.6, 0.05,
            help="How confident the model must be before it agrees a hand is present "
                 "at all. Lower this if your hand isn't being detected (e.g. in dim "
                 "lighting); raise it if it's detecting hands that aren't really there.",
        ),
        tracking_confidence=st.sidebar.slider(
            "Tracking confidence", 0.1, 1.0, 0.6, 0.05,
            help="How confident the model must be to keep following an already-"
                 "detected hand between frames. Lower this if tracking keeps dropping "
                 "out during fast movement; raise it for steadier, more conservative tracking.",
        ),
        show_fps=st.sidebar.checkbox(
            "Show FPS counter", value=True,
            help="Displays how many frames per second are being processed, "
                 "in the top-left corner of the video.",
        ),
        show_finger_count=st.sidebar.checkbox(
            "Show finger count", value=True,
            help="Displays how many fingers are raised on each detected hand, "
                 "labeled with which hand (Left/Right) it belongs to.",
        ),
        draw_style=st.sidebar.radio(
            "Landmark style", ["Skeleton", "Dots only"], index=0,
            help="'Skeleton' draws the 21 hand points connected by lines, showing the "
                 "hand's structure. 'Dots only' shows just the points, with no lines.",
        ),
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Tip: allow camera access when your browser prompts you. "
        "Nothing is recorded — frames are processed live in memory and discarded."
    )
    return settings


def render_about() -> None:
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
            "- Each frame is streamed over WebRTC, processed **live**, and sent back\n"
            "- Finger count uses landmark geometry, no extra model\n"
            "- Nothing is recorded or stored"
        )

    with col2:
        st.subheader("Built with")
        st.markdown("`OpenCV` · `MediaPipe` · `Streamlit` · `streamlit-webrtc`")

    st.caption(
        "Original concept inspired by open-source OpenCV/MediaPipe hand-tracking scripts, "
        "rebuilt from the ground up for live browser deployment."
    )
