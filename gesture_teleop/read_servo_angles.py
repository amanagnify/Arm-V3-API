#!/usr/bin/env python3
from __future__ import annotations

"""Read V3 firmware STATE lines from the board USB-UART.

The Zephyr firmware logs rate-limited absolute servo targets like:
    STATE: base=90.0 p1=98.1 p2=88.2 p3=90.0 roll=90.0 grip=166.5 mode=M seq=12
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator


STATE_RE = re.compile(
    r"STATE:\s+"
    r"base=(?P<base>[+-]?\d+(?:\.\d+)?)\s+"
    r"p1=(?P<p1>[+-]?\d+(?:\.\d+)?)\s+"
    r"p2=(?P<p2>[+-]?\d+(?:\.\d+)?)\s+"
    r"p3=(?P<p3>[+-]?\d+(?:\.\d+)?)\s+"
    r"roll=(?P<roll>[+-]?\d+(?:\.\d+)?)\s+"
    r"grip=(?P<grip>[+-]?\d+(?:\.\d+)?)\s+"
    r"mode=(?P<mode>[AHM])\s+"
    r"seq=(?P<seq>\d+)"
)


@dataclass
class ServoState:
    base: float = 90.0
    p1: float = 98.1
    p2: float = 88.2
    p3: float = 90.0
    roll: float = 90.0
    grip: float = 166.5
    mode: str = "?"
    seq: int = 0
    last_update: float = field(default_factory=time.perf_counter)


def parse_state_line(line: str, state: ServoState) -> bool:
    match = STATE_RE.search(line)
    if not match:
        return False

    state.base = float(match.group("base"))
    state.p1 = float(match.group("p1"))
    state.p2 = float(match.group("p2"))
    state.p3 = float(match.group("p3"))
    state.roll = float(match.group("roll"))
    state.grip = float(match.group("grip"))
    state.mode = match.group("mode")
    state.seq = int(match.group("seq"))
    state.last_update = time.perf_counter()
    return True


def _serial_lines(port: str, baud: int) -> Iterator[str]:
    try:
        import serial  # type: ignore
    except ImportError:
        print("ERROR: pyserial is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    print(f"Opening serial port {port} at {baud} baud...", flush=True)
    with serial.Serial(port, baud, timeout=0.5) as ser:
        ser.reset_input_buffer()
        buf = b""
        while True:
            chunk = ser.read(256)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf or b"\r" in buf:
                idx_n = buf.find(b"\n")
                idx_r = buf.find(b"\r")
                if idx_n != -1 and (idx_r == -1 or idx_n < idx_r):
                    line_bytes, buf = buf.split(b"\n", 1)
                else:
                    line_bytes, buf = buf.split(b"\r", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if line:
                    yield line


def _auto_detect_port() -> str:
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError:
        print("ERROR: pyserial is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    all_ports = list(list_ports.comports())
    keywords = ("zephyr", "esp32", "uart", "usb serial", "usb-serial", "acm", "jtag")
    for port in all_ports:
        haystack = f"{port.device} {port.description or ''} {port.manufacturer or ''}".lower()
        if any(keyword in haystack for keyword in keywords):
            return port.device

    if all_ports:
        return all_ports[0].device

    print("ERROR: no serial ports found. Use --list-ports to inspect devices.", file=sys.stderr)
    sys.exit(1)


def _list_ports() -> None:
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError:
        print("ERROR: pyserial is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    print("Available serial ports:")
    for port in list_ports.comports():
        print(f"  {port.device:16s} {port.description} [mfr: {port.manufacturer}]")


def _render(state: ServoState, source: str, stale_s: float) -> str:
    age = time.perf_counter() - state.last_update
    stale = f" STALE {age:.1f}s" if age > stale_s else ""
    rows = (
        ("base", state.base, "0..180"),
        ("p1", state.p1, "0..100"),
        ("p2", state.p2, "45..180"),
        ("p3", state.p3, "0..180"),
        ("roll", state.roll, "0..180"),
        ("grip", state.grip, "36..166.5"),
    )
    lines = [
        "\033[2J\033[H",
        f"Robot Arm V3 servo targets from {source}{stale}",
        f"mode={state.mode} seq={state.seq}",
        "",
    ]
    for name, value, limits in rows:
        lines.append(f"{name:>5}: {value:6.1f} deg   limit {limits}")
    lines.append("")
    lines.append("Waiting for firmware STATE lines. Press Ctrl-C to exit.")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read Robot Arm V3 firmware STATE lines.")
    parser.add_argument("--port", help="Serial port, for example COM5 or /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--stale-s", type=float, default=1.5)
    parser.add_argument("--list-ports", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list_ports:
        _list_ports()
        return 0

    port = args.port or _auto_detect_port()
    source = f"serial {port}"
    state = ServoState()

    try:
        for line in _serial_lines(port, args.baud):
            updated = parse_state_line(line, state)
            print(_render(state, source, args.stale_s), end="", flush=True)
            if not updated and "STATE:" not in line:
                print(f"\n[{line[:100]}]", flush=True)
    except KeyboardInterrupt:
        print("\nExiting.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
