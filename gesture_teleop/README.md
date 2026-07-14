# 6-DOF Robotic Arm: Gesture Teleoperation & API

This project contains the Python codebase to control a 6-DOF robotic arm using a camera (MediaPipe hand tracking) or via a fully standalone REST API.

## 🚀 Architecture Overview

We recently upgraded the architecture to use a **FastAPI Middleware Backend**.

Previously, the OpenCV/MediaPipe script (`main.py`) talked directly to the ESP32. Now, it is fully decoupled:
1. **The Brain (`api/main.py`)**: A fast, asynchronous web server that manages the WebSocket connection to the robot. It queues commands safely, enforces servo limits, and auto-reconnects.
2. **The Eyes (`main.py`)**: The camera tracking script now simply sends lightweight `HTTP POST` requests to the API instead of maintaining a complex stateful connection.

## 🛠️ How to Use

### 1. Start the API Server (The Brain)
You must start the API server first. It will connect to the robot and wait for commands.
```bash
cd api/
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```
*Wait until you see `WebSocket client connected` in the terminal.*

### 2. Start the Camera (The Eyes)
In a **new terminal**, start the hand-tracking script and tell it to use the `http` transport so it talks to the API:
```bash
python main.py --transport http
```

### 3. Use the Web Dashboard (No Camera Needed!)
You can manually move the robot, check its status, or automate it using any language. 
Open your browser and navigate to:
**👉 http://127.0.0.1:8000/docs**

Here you will find a sleek Dark Mode UI where you can test moving the robot via the `POST /robot/move` endpoint without ever turning on your camera.

---

## 🗑️ Clean Up & Deprecated Files
The legacy direct-connection scripts have been deprecated and safely unlinked from the code. If you still see the following files in your folder, you can safely delete them as they are no longer part of the project:
- ❌ `tcp_comm.py`
- ❌ `websocket_comm.py` 
*(All transport logic is now handled beautifully by `api/services/websocket_service.py` and `http_comm.py`)*
