"""OpenCV overlays drawn onto video frames."""
import cv2
import mediapipe as mp

from .config import AMBER_COLOR, SIGNAL_COLOR

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand(img, points, style: str) -> None:
    """Draws landmarks as a connected skeleton or as dots only."""
    if style == "Skeleton":
        for conn in HAND_CONNECTIONS:
            cv2.line(img, points[conn.start], points[conn.end], SIGNAL_COLOR, 2)
        radius = 4
    else:
        radius = 5
    for point in points:
        cv2.circle(img, point, radius, SIGNAL_COLOR, cv2.FILLED)


def draw_finger_count(img, wrist, label: str, count: int) -> None:
    """Writes e.g. 'Left: 3' just below the wrist."""
    wrist_x, wrist_y = wrist
    position = (max(wrist_x - 30, 10), max(wrist_y + 40, 30))
    cv2.putText(
        img, f"{label}: {count}", position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER_COLOR, 2, cv2.LINE_AA,
    )


def draw_fps(img, fps: float) -> None:
    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, SIGNAL_COLOR, 2, cv2.LINE_AA,
    )
