# Real-Time Hand Tracking (Browser-Deployed)

A live, browser-based hand-tracking app built with **OpenCV**, **MediaPipe**, and **Streamlit**, using **streamlit-webrtc** to stream and process your webcam feed directly in the browser — no local install needed, no server-side camera required.

This project is a rebuild of a classic OpenCV + MediaPipe hand-tracking script, redesigned specifically to run as a public, clickable web app rather than a local-only Python script.

## Why a rebuild ?

Two separate problems made a direct deployment approach impossible:

1. **No server-side camera.** The original approach uses `cv2.VideoCapture(0)`, which opens *the machine's own* webcam in a local window — fine on a laptop, but impossible on a cloud server with no physical camera. The fix: `streamlit-webrtc` streams the **visitor's own browser webcam** over WebRTC, processes each frame live with MediaPipe, and streams the annotated result back — no video ever touches disk or leaves the session.
2. **A removed API.** MediaPipe's older `mp.solutions.hands` interface (used in most tutorials and older projects) has been removed in current MediaPipe releases. This app is built on the actively maintained `mediapipe.tasks` `HandLandmarker` API instead, with the required model file fetched automatically on first run (see `ensure_model_downloaded()` in `app.py`) so there's no manual asset step.

## Features

- Real-time multi-hand landmark detection (up to 4 hands)
- Left/right hand classification
- Live finger-count overlay per detected hand
- Adjustable detection/tracking confidence thresholds (sidebar)
- Two landmark rendering styles (full skeleton or dots only)
- Live FPS counter
- Fully client-camera-driven — no footage stored or transmitted anywhere

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`) and allow camera access when prompted.

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to its own GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app," select this repo, set the main file to `app.py`, and deploy.
4. You'll get a live URL like `yourapp.streamlit.app` — that's the link to put in your portfolio.

## Tech stack

`Python` · `OpenCV` · `MediaPipe` · `Streamlit` · `streamlit-webrtc`

## Credit

Concept inspired by open-source OpenCV/MediaPipe hand-tracking scripts in the computer vision community; implementation, browser-deployment architecture, and feature additions (finger counting, adjustable settings, multi-hand labeling) built independently for this project.