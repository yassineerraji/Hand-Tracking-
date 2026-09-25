"""Streamlit entrypoint: `streamlit run app.py`."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ice_servers():
    # Credentials live in .streamlit/secrets.toml locally, or in the
    # Streamlit Cloud app's Secrets settings when deployed.
    try:
        key_id = st.secrets.get("CF_TURN_KEY_ID")
        api_token = st.secrets.get("CF_TURN_API_TOKEN")
    except FileNotFoundError:  # no secrets configured at all
        key_id = api_token = None
    return get_ice_servers(key_id, api_token)


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
