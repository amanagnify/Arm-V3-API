# FastAPI Middleware Migration Guide

## Architecture Overview

The new architecture introduces a FastAPI middleware layer between the Python vision processing application and the ESP32 robotic controller.

**Old Flow:**
`vision.py` → `transport.py` → `websocket_comm.py`/`tcp_comm.py` → ESP32

**New Flow:**
`vision.py` → `transport.py` (`http_comm.py`) → FastAPI → `robot_service.py` → `websocket_service.py` → ESP32

## Migration Steps

1. **Install Requirements:**
   Navigate to the `gesture_teleop/api` folder and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure FastAPI ESP32 URL:**
   In `gesture_teleop/api/config.py`, verify or modify the `ESP32_WS_URL` to match your ESP32's IP and WebSocket port (e.g. `ws://192.168.1.100:81`).

3. **Start the FastAPI Backend:**
   From the `gesture_teleop` folder, run the following command to start the FastAPI server on port 8000:
   ```bash
   uvicorn api.main:app --host 0.0.0.1 --port 8000
   ```

4. **Update Application Configuration:**
   Update your `gesture_teleop/config/calibration.json` (or whichever config you use) to utilize the new `http` transport.
   Add or update the transport section:
   ```json
   "transport": {
       "enabled": true,
       "kind": "http"
   },
   "http_host": "127.0.0.1",
   "http_port": 8000
   ```
   Or use command line arguments if you modify `main.py` arguments. 
   *(Alternatively, run `python main.py --transport http` if you add the argument in `main.py` or rely on the config).*

5. **Run the Vision Pipeline:**
   Start the control pipeline normally:
   ```bash
   python main.py --transport http
   ```

## Example HTTP Requests

You can test the FastAPI server directly with these examples:

### Post Gesture Command
```bash
curl -X POST "http://127.0.0.1:8000/gesture" \
     -H "Content-Type: application/json" \
     -d '{
           "joint1": 90,
           "joint2": 45,
           "joint3": 120,
           "joint4": 60,
           "joint5": 30,
           "gripper": 15,
           "mode": "T"
         }'
```

### Move Robot Manually
```bash
curl -X POST "http://127.0.0.1:8000/robot/move" \
     -H "Content-Type: application/json" \
     -d '{
           "joint1": 100,
           "joint2": 50,
           "joint3": 110,
           "joint4": 60,
           "joint5": 30,
           "gripper": 100,
           "mode": "M"
         }'
```

### Get Robot Status
```bash
curl -X GET "http://127.0.0.1:8000/robot/status"
```

### Check Health
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
