"""Static configuration: paths, URLs, colors and defaults."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The MediaPipe Tasks API needs a .task model file. It is committed under
# models/, and downloaded from MODEL_URL if it's ever missing.
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Public STUN server, used alone when no TURN credentials are configured.
STUN_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
CLOUDFLARE_TURN_URL = (
    "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}"
    "/credentials/generate-ice-servers"
)
TURN_CREDENTIAL_TTL_S = 86400

# Requested camera resolution; kept modest so shared cloud CPUs keep up.
VIDEO_CONSTRAINTS = {"width": {"ideal": 640}, "height": {"ideal": 480}}

# OpenCV colors are BGR.
SIGNAL_COLOR = (212, 234, 94)   # #5EEAD4
AMBER_COLOR = (90, 166, 242)    # #F2A65A


@dataclass(frozen=True)
class Settings:
    """User-tunable options, set from the sidebar and read by the processor."""
    max_hands: int = 2
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    show_fps: bool = True
    show_finger_count: bool = True
    draw_style: str = "Skeleton"  # "Skeleton" or "Dots only"
