"""Per-frame video processing: MediaPipe detection plus overlays."""
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp

from .config import Settings
from .drawing import draw_finger_count, draw_fps, draw_hand
from .gestures import count_raised_fingers, viewer_handedness

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


class HandTrackingProcessor:
    """Wraps a MediaPipe Tasks HandLandmarker for per-frame video processing.

    One instance lives per WebRTC session. `settings` is replaced from the
    Streamlit script thread whenever the sidebar changes; recv() runs on a
    separate worker thread and rebuilds the landmarker lazily if the
    model-level options (hands, confidences) differ from what it was built with.
    """

    def __init__(self, model_path: Path, settings: Settings = Settings()):
        self.model_path = model_path
        self.settings = settings
        self._landmarker = None
        self._built_with = None
        self._start_time = time.time()
        self._prev_time = self._start_time

    def _ensure_landmarker(self, settings: Settings):
        wanted = (settings.max_hands, settings.detection_confidence, settings.tracking_confidence)
        if self._landmarker is not None and self._built_with == wanted:
            return
        if self._landmarker is not None:
            self._landmarker.close()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=settings.detection_confidence,
            min_tracking_confidence=settings.tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._built_with = wanted

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        settings = self.settings  # read once so a mid-frame update can't mix values
        self._ensure_landmarker(settings)

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror for a natural "looking in a mirror" feel
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            model_label = "Right"
            if idx < len(result.handedness) and result.handedness[idx]:
                model_label = result.handedness[idx][0].category_name or "Right"

            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            draw_hand(img, points, settings.draw_style)

            if settings.show_finger_count:
                count = count_raised_fingers(hand_landmarks)
                draw_finger_count(img, points[0], viewer_handedness(model_label), count)

        if settings.show_fps:
            now = time.time()
            fps = 1 / (now - self._prev_time) if now != self._prev_time else 0
            self._prev_time = now
            draw_fps(img, fps)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
