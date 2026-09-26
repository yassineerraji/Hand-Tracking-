# ✋ Real-Time Hand Tracking

**[▶ Live demo](LIVE_DEMO_URL)**: open it, allow camera access, and show your hands.

A real-time computer vision web app that detects hands in your webcam feed, tracks 21 keypoints per hand, identifies left from right, and counts raised fingers live in the browser.

## Highlights

- **Real-time hand pose estimation** with Google's MediaPipe HandLandmarker: 21 3D keypoints per hand, up to 4 hands at once
- **Rule-based finger counting** from keypoint geometry, with no extra model. The thumb test uses distances instead of x-coordinates, so it works on either hand and in a mirrored view
- **Live video pipeline**: the webcam streams over WebRTC, each frame is processed server-side and the annotated video comes straight back. Nothing is recorded or stored
- **Interactive controls** for detection and tracking confidence, number of hands, and overlay style, applied without restarting the stream
- **Production-ready**: deployed on Streamlit Cloud with a TURN relay for reliable connections, modular code and unit tests

## Tech stack

`Python` · `MediaPipe` · `OpenCV` · `NumPy` · `Streamlit` · `WebRTC` · `pytest`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Project structure

```
app.py              Streamlit entrypoint
hand_tracking/      Detection, gesture logic, drawing, video processing, UI
models/             MediaPipe hand landmark model
tests/              Unit tests
```
