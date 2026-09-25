# Real-Time Hand Tracking

A live hand-tracking web app built with **MediaPipe**, **OpenCV** and **Streamlit**. It streams your webcam to the app over WebRTC (via **streamlit-webrtc**), detects hand landmarks on every frame, and streams the annotated video straight back to your browser. There's nothing to install: open the page and allow camera access.

This project is a rebuild of a classic OpenCV + MediaPipe hand-tracking script, redesigned to run as a public web app instead of a local-only Python script.

## Features

- Real-time multi-hand landmark detection (up to 4 hands)
- Left/right hand classification
- Live finger-count overlay per detected hand
- Adjustable detection and tracking confidence thresholds
- Two landmark styles: full skeleton or dots only
- Live FPS counter
- Privacy-friendly: frames are processed in memory and discarded, nothing is recorded or stored

## How it works

The original script had two problems that ruled out deploying it as-is:

1. **No server-side camera.** `cv2.VideoCapture(0)` opens the webcam of the machine running the code. That works on a laptop, but a cloud server has no camera. Here, `streamlit-webrtc` streams the **visitor's own webcam** from the browser to the app instead. Each frame is processed live with MediaPipe, and the annotated result is sent back.
2. **A removed API.** MediaPipe's legacy `mp.solutions.hands` interface, used in most tutorials, no longer exists in current releases. This app uses the maintained `mediapipe.tasks` `HandLandmarker` API instead. Its model file ships in `models/` and is re-downloaded automatically if it's missing.

Finger counting uses landmark geometry alone, with no extra model. See `hand_tracking/gestures.py`.

## Project structure

```
app.py                  Streamlit entrypoint: wires the modules together
hand_tracking/
  config.py             Paths, colors, defaults and the Settings dataclass
  model.py              Locates / downloads the MediaPipe model
  gestures.py           Finger counting and handedness (pure logic)
  drawing.py            OpenCV overlays (skeleton, labels, FPS)
  processor.py          Per-frame WebRTC video processor
  ice.py                STUN/TURN server configuration (Cloudflare)
  ui.py                 Sidebar controls and page content
models/                 MediaPipe hand landmarker model
tests/                  pytest suite
requirements.txt        Runtime dependencies (pinned)
requirements-dev.txt    Runtime + test dependencies
packages.txt            System libraries installed by Streamlit Cloud
```

## Run locally

Requires Python 3.10 or newer.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Open http://localhost:8501 and allow camera access. Run the tests with `pytest`.

## Deploy (Streamlit Community Cloud, free)

Cloud servers sit behind NATs that block direct WebRTC connections, so the app needs a **TURN relay** in production (STUN alone only works locally). It uses Cloudflare's free TURN service:

1. In the Cloudflare dashboard, go to **Realtime → TURN Server → Create**, then copy the *Turn Token ID* and the *API Token*.
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app**. Pick this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 and paste this into **Secrets**:
   ```toml
   CF_TURN_KEY_ID = "your-turn-token-id"
   CF_TURN_API_TOKEN = "your-api-token"
   ```
4. Click **Deploy**.

To use TURN locally, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (it's git-ignored) and fill in the same two values. Without them, the app falls back to STUN only, which works on localhost.

## Tech stack

`Python` · `MediaPipe` · `OpenCV` · `Streamlit` · `streamlit-webrtc` · `Cloudflare TURN`

## Credit

Concept inspired by open-source OpenCV/MediaPipe hand-tracking scripts in the computer vision community. The implementation, browser-deployment architecture and added features (finger counting, adjustable settings, multi-hand labeling) were built independently for this project.
