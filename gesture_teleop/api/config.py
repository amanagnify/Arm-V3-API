import os

# Application Settings
ESP32_WS_URL = os.getenv("ESP32_WS_URL", "ws://192.168.1.30:4220") # Default ESP32 WebSocket URL
WS_RECONNECT_INTERVAL = 3.0 # seconds

# Safety Limits
SERVO_MIN = 0.0
SERVO_MAX = 180.0
