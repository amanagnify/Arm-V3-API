import asyncio
import websockets
import logging
from typing import Optional, Callable
from api.config import ESP32_WS_URL, WS_RECONNECT_INTERVAL
from api.models.robot_models import RobotStatus

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.url = ESP32_WS_URL
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self.status = RobotStatus(connected=False)
        self.queue = asyncio.Queue()
        self._reconnect_task = None
        self._send_task = None

    async def connect(self):
        self._running = True
        self._reconnect_task = asyncio.create_task(self._maintain_connection())
        self._send_task = asyncio.create_task(self._send_loop())

    async def disconnect(self):
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._send_task:
            self._send_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _maintain_connection(self):
        while self._running:
            if not self._connected:
                try:
                    logger.info(f"Connecting to ESP32 at {self.url}...")
                    self.ws = await websockets.connect(self.url, ping_interval=None)
                    self._connected = True
                    self.status.connected = True
                    logger.info("Connected to ESP32 WebSocket")
                    await self._receive_loop()
                except Exception as e:
                    logger.error(f"WebSocket connection failed: {e}")
                    self._connected = False
                    self.status.connected = False
                    await asyncio.sleep(WS_RECONNECT_INTERVAL)
            else:
                await asyncio.sleep(1)

    async def _receive_loop(self):
        try:
            while self._running and self._connected and self.ws:
                message = await self.ws.recv()
                # Assuming the ESP32 sends status back
                # self._update_status(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        finally:
            self._connected = False
            self.status.connected = False
            if self.ws:
                await self.ws.close()
                self.ws = None

    async def _send_loop(self):
        # Default safe parked pose to send as keep-alive before the first HTTP request
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
        try:
            import protocol
            last_packet = protocol.build_packet(0, "M", 90.0, 45.0, 112.5, 90.0, 90.0, 166.5)
        except Exception:
            last_packet = b"" # Fallback

        while self._running:
            try:
                packet = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                last_packet = packet
                self.queue.task_done()
            except asyncio.TimeoutError:
                packet = last_packet

            if self._connected and self.ws and packet:
                try:
                    # Timeout ensures if WiFi is dead, it throws error and disconnects
                    await asyncio.wait_for(self.ws.send(packet), timeout=0.5)
                except Exception as e:
                    logger.error(f"Error sending packet (WiFi drop?): {e}")
                    self._connected = False
                    if self.ws:
                        try:
                            await self.ws.close()
                        except Exception:
                            pass
                        self.ws = None

    async def send_packet(self, packet: bytes):
        # We can queue it or send it directly. Queueing is safer for preventing multiple writes
        if self.queue.qsize() > 5: # prevent buildup
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.queue.put(packet)

ws_manager = WebSocketManager()
