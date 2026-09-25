"""Gesture logic computed from MediaPipe's 21 hand landmarks."""
import math

THUMB_TIP, THUMB_MCP, PINKY_MCP = 4, 2, 17
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky


def count_raised_fingers(landmarks) -> int:
    """Counts extended fingers from 21 hand landmarks (objects with .x/.y).

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
        return math.hypot(a.x - b.x, a.y - b.y)

    fingers_up = 0

    if dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) > dist(landmarks[THUMB_MCP], landmarks[PINKY_MCP]):
        fingers_up += 1

    for tip_id in FINGER_TIPS:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers_up += 1

    return fingers_up


def viewer_handedness(model_label: str) -> str:
    """Converts MediaPipe's handedness label to the one the viewer perceives.

    We feed MediaPipe the mirrored frame so the live view looks natural,
    which means its Left/Right label is the mirror image of what the viewer
    perceives as their own hand. Swap it back.
    """
    return "Right" if model_label == "Left" else "Left"
