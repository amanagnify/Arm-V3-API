from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from config import AppConfig, load_config, save_config

if TYPE_CHECKING:
    from vision import HandObservation


def capture_neutral_reference(config: AppConfig, observation: HandObservation) -> None:
    """Capture the current hand as the neutral reference. Press 'n' with an
    OPEN palm: the current finger curl doubles as the grip OPEN reference and
    the closed reference is derived as open + grip_close_span."""
    config.vision.neutral_x = observation.center_xy[0]
    config.vision.neutral_y = observation.center_xy[1]
    config.vision.depth_reference = observation.depth_metric
    config.vision.wrist_tilt_reference = observation.wrist_tilt_metric
    config.vision.grip_open_reference = observation.finger_curl_metric
    config.vision.grip_closed_reference = (
        observation.finger_curl_metric + config.vision.grip_close_span
    )


def capture_pinch_reference(config: AppConfig, observation: HandObservation, open_reference: bool) -> None:
    if open_reference:
        config.vision.pinch_open_reference = observation.pinch_metric
    else:
        config.vision.pinch_closed_reference = observation.pinch_metric


def capture_grip_reference(config: AppConfig, observation: HandObservation, open_reference: bool) -> None:
    if open_reference:
        config.vision.grip_open_reference = observation.finger_curl_metric
    else:
        config.vision.grip_closed_reference = observation.finger_curl_metric


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration helper utilities.")
    parser.add_argument("--config", default="config/calibration.json", help="Path to calibration JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show", help="Print the current config path.")

    neutral = subparsers.add_parser("set-neutral", help="Set neutral hand reference.")
    neutral.add_argument("--x", type=float, required=True)
    neutral.add_argument("--y", type=float, required=True)
    neutral.add_argument("--depth", type=float, required=True)

    pinch = subparsers.add_parser("set-pinch", help="Set pinch calibration values.")
    pinch.add_argument("--open", dest="pinch_open", type=float)
    pinch.add_argument("--closed", dest="pinch_closed", type=float)

    grip = subparsers.add_parser("set-grip", help="Set grip curl calibration values.")
    grip.add_argument("--open", dest="grip_open", type=float)
    grip.add_argument("--closed", dest="grip_closed", type=float)

    region = subparsers.add_parser("set-region", help="Set teleop active region.")
    region.add_argument("--x-min", type=float, required=True)
    region.add_argument("--y-min", type=float, required=True)
    region.add_argument("--x-max", type=float, required=True)
    region.add_argument("--y-max", type=float, required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    if args.command == "show":
        print(config.config_path)
        return

    if args.command == "set-neutral":
        config.vision.neutral_x = args.x
        config.vision.neutral_y = args.y
        config.vision.depth_reference = args.depth
        save_config(config)
        return

    if args.command == "set-pinch":
        if args.pinch_open is not None:
            config.vision.pinch_open_reference = args.pinch_open
        if args.pinch_closed is not None:
            config.vision.pinch_closed_reference = args.pinch_closed
        save_config(config)
        return

    if args.command == "set-grip":
        if args.grip_open is not None:
            config.vision.grip_open_reference = args.grip_open
        if args.grip_closed is not None:
            config.vision.grip_closed_reference = args.grip_closed
        save_config(config)
        return

    if args.command == "set-region":
        config.vision.active_region.x_min = args.x_min
        config.vision.active_region.y_min = args.y_min
        config.vision.active_region.x_max = args.x_max
        config.vision.active_region.y_max = args.y_max
        save_config(config)
        return


if __name__ == "__main__":
    main()
