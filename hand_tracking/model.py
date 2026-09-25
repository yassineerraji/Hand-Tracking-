"""Locating (and if necessary downloading) the MediaPipe hand landmarker model."""
import urllib.request
from pathlib import Path

from .config import MODEL_PATH, MODEL_URL


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Returns the model path, downloading the file first if it's missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path
