"""Streamlit entrypoint: `streamlit run app.py`."""
import os

import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from hand_tracking.config import VIDEO_CONSTRAINTS
from hand_tracking.ice import get_ice_servers
from hand_tracking.model import ensure_model
from hand_tracking.processor import HandTrackingProcessor
from hand_tracking.ui import render_about, render_sidebar

st.set_page_config(
    page_title="Real-Time Hand Tracking",
    page_icon="✋",
    layout="wide",
)


@st.cache_resource(show_spinner="Downloading hand-tracking model (first run only)...")
def cached_model_path():
    return ensure_model()


def read_secret(name):
    """Reads a secret from the environment (Hugging Face Space secrets, Docker)
    or from .streamlit/secrets.toml (local runs, Streamlit Cloud)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except FileNotFoundError:  # no secrets.toml at all
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ice_servers():
    return get_ice_servers(read_secret("CF_TURN_KEY_ID"), read_secret("CF_TURN_API_TOKEN"))


settings = render_sidebar()

st.title("✋ Real-Time Hand Tracking")

model_path = cached_model_path()

ctx = webrtc_streamer(
    key="hand-tracking",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": cached_ice_servers()}),
    video_processor_factory=lambda: HandTrackingProcessor(model_path, settings),
    media_stream_constraints={"video": VIDEO_CONSTRAINTS, "audio": False},
    async_processing=True,
)

# Push the current sidebar values into the running processor (if active)
# so changes take effect live without restarting the stream.
if ctx.video_processor:
    ctx.video_processor.settings = settings

render_about()
