from hand_tracking.config import MODEL_PATH
from hand_tracking.model import ensure_model


def test_bundled_model_is_found_without_download():
    assert ensure_model() == MODEL_PATH
    assert MODEL_PATH.stat().st_size > 1_000_000
