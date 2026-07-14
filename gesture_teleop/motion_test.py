#!/usr/bin/env python3
"""Repeatable large-move test sequence for p1/p2/p3 smoothness evaluation.

Cycles ACTIVE -> REACH_UP -> TUCK_DOWN -> ACTIVE with big pitch sweeps
(p1 ~85 deg, p2 ~100 deg, p3 ~120 deg) so profile changes can be compared
one at a time on the real arm. No camera needed.

Two host modes (combine with the firmware PROFILE_MODE for the A/B matrix):
  step  — send each pose as a raw step target (streamed at 30 Hz): the
          FIRMWARE does all smoothing. Tests firmware-only smoothing.
  ramp  — host streams a minimum-jerk trajectory at --rate deg/s peak:
          tests host-led / hybrid smoothing.

Examples:
  python motion_test.py --host 192.168.1.33 --mode step
  python motion_test.py --host 192.168.1.33 --mode ramp --rate 15 --cycles 3
  python motion_test.py --dry-run --mode ramp        # offline sanity check

While it runs, watch the firmware MOTION:/STATE: lines on the serial monitor
(python read_servo_angles.py) and judge: smooth start? continuous mid-move
(no accel-slow-accel)? smooth stop? overshoot? do p1/p2/p3 arrive together?
"""
from __future__ import annotations

import argparse
import sys
import time

from config import load_config
from main import Pose, PoseTransition
from transport import build_transport

SEND_HZ = 30.0

# p2 kept consistent with the mounted-flipped convention: p2 = 180 - avg(p1, p3)
POSES = (
    ("ACTIVE", Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)),
    ("REACH_UP", Pose(90.0, 90.0, 60.0, 150.0, 90.0, 1.0)),
    ("TUCK_DOWN", Pose(90.0, 5.0, 162.5, 30.0, 90.0, 1.0)),
    ("ACTIVE", Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Large p1/p2/p3 move sequence (V3).")
    p.add_argument("--transport", choices=("websocket", "tcp"), default="websocket")
    p.add_argument("--host")
    p.add_argument("--port", type=int, default=4210)
    p.add_argument("--mode", choices=("step", "ramp"), default="ramp",
                   help="step: firmware does all smoothing; ramp: host min-jerk")
    p.add_argument("--rate", type=float, default=10.0,
                   help="ramp mode: peak trajectory speed in deg/s (default 10)")
    p.add_argument("--dwell", type=float, default=2.5,
                   help="seconds to hold at each pose (default 2.5)")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--dry-run", action="store_true", help="no network; just print")
    return p.parse_args()


def _stream(transport, pose: Pose, gripper_cfg, seconds: float) -> None:
    """Stream the (constant) pose at SEND_HZ for `seconds`."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        transport.send(pose.command("A", gripper_cfg), force=True)
        time.sleep(1.0 / SEND_HZ)


def main() -> int:
    args = _parse_args()
    config = load_config("config/calibration.json")

    config.transport.enabled = True
    config.transport.kind = args.transport
    tcfg = config.tcp if args.transport == "tcp" else config.websocket
    tcfg.enabled = True
    tcfg.dry_run = args.dry_run
    if args.host:
        tcfg.host = args.host
    tcfg.port = args.port

    grip_span = max(abs(config.gripper.closed_deg - config.gripper.open_deg), 1e-6)
    transport = build_transport(config)
    transport.connect()
    if not transport.connected:
        print(f"Could not connect ({transport.last_error}). Use --host <esp32-ip> "
              "or --dry-run.", file=sys.stderr)
        return 1

    pose = Pose(*[getattr(POSES[0][1], f) for f in ("base", "p1", "p2", "p3", "roll", "grip")])
    print(f"mode={args.mode} rate={args.rate} deg/s dwell={args.dwell}s "
          f"cycles={args.cycles} dry_run={args.dry_run}")
    print("settling at ACTIVE...")
    _stream(transport, pose, config.gripper, 2.0)

    try:
        for cycle in range(1, args.cycles + 1):
            for name, target in POSES[1:]:
                deltas = [abs(getattr(target, f) - getattr(pose, f))
                          for f in ("p1", "p2", "p3")]
                print(f"[cycle {cycle}] -> {name}  "
                      f"(dp1={deltas[0]:.0f} dp2={deltas[1]:.0f} dp3={deltas[2]:.0f} deg)")

                if args.mode == "step":
                    pose.set_from(target)
                    travel = max(deltas) / 35.0 + 1.0  # rough firmware travel time
                    _stream(transport, pose, config.gripper, travel)
                else:
                    tr = PoseTransition(pose, target, args.rate, grip_span)
                    print(f"           min-jerk duration {tr.duration:.1f}s")
                    last = time.perf_counter()
                    done = False
                    while not done:
                        now = time.perf_counter()
                        done = tr.step(pose, now - last)
                        last = now
                        transport.send(pose.command("A", config.gripper), force=True)
                        time.sleep(1.0 / SEND_HZ)

                _stream(transport, pose, config.gripper, args.dwell)
        print("sequence complete; holding last pose")
        transport.send(pose.command("H", config.gripper), force=True)
    except KeyboardInterrupt:
        print("\ninterrupted; sending hold")
        transport.send(pose.command("H", config.gripper), force=True)
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
