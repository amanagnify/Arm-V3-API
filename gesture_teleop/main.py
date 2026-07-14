from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from calibration import capture_neutral_reference
from config import AppConfig, GripperConfig, load_config, save_config
from filters import (
    AccelSlewLimiter,
    ExpSmoothLimiter,
    SCurveLimiter,
    SlewRateLimiter,
    clamp,
    make_input_filter,
    min_jerk_s,
)
from hand_mapping import (
    compute_base_angle,
    compute_pitch_targets,
    compute_roll_angle,
    extract_signals,
    integrate_rate,
    reach_height_to_rz,
)
from ps3_joystick import JoystickDriver, JoystickState, probe_joystick, shutdown_joystick
from transport import TeleopCommand, build_transport
from two_hand_control import TwoHandController
from ui_overlay import TeleopUI
from vision import HandTracker, best_of

WINDOW_NAME = "Hand Teleop (6-DOF)"

# cv2.waitKeyEx arrow codes (Windows first, X11 second)
KEY_UP = (2490368, 65362)
KEY_DOWN = (2621440, 65364)


@dataclass
class Pose:
    base: float
    p1: float
    p2: float
    p3: float
    roll: float
    grip: float  # open fraction [0,1]; converted to a servo angle on send

    def command(self, mode: str, gripper_cfg: GripperConfig) -> TeleopCommand:
        open_deg = gripper_cfg.open_deg
        closed_deg = gripper_cfg.closed_deg
        grip_deg = open_deg + (1.0 - clamp(self.grip, 0.0, 1.0)) * (closed_deg - open_deg)
        return TeleopCommand(
            mode=mode,
            base_deg=self.base,
            lower_deg=self.p1,
            middle_deg=self.p2,
            upper_deg=self.p3,
            wrist_deg=self.roll,
            gripper_deg=grip_deg,
        )

    def set_from(self, other: "Pose") -> None:
        self.base, self.p1, self.p2 = other.base, other.p1, other.p2
        self.p3, self.roll, self.grip = other.p3, other.roll, other.grip


@dataclass
class ControlState:
    """All the mutable per-frame control state (hold dwell timers, pre-gesture
    snapshots, IK continuity, last-computed reach) lives here so the main loop
    stays flat."""
    grip_hold_active: bool = False
    grip_hold_ready: bool = True
    grip_hold_start: float = 0.0
    grip_above_since: float | None = None
    grip_depth_snapshot: float | None = None
    roll_above_since: float | None = None
    roll_depth_snapshot: float | None = None
    roll_height_snapshot: float | None = None
    depth_hold_prev: bool = False
    roll_active_prev: bool = False
    held_depth: float = 0.5
    held_height: float = 0.5
    prev_pitch: tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_hold: bool = False
    roll_active: bool = False
    reach_r: float = 0.0
    reach_z: float = 0.0

    def reset_locks(self) -> None:
        self.grip_hold_active = False
        self.grip_above_since = None
        self.grip_depth_snapshot = None
        self.roll_above_since = None
        self.roll_depth_snapshot = None
        self.roll_height_snapshot = None
        self.depth_hold_prev = False
        self.roll_active_prev = False
        self.depth_hold = False
        self.roll_active = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-pose teleop for a 6-servo arm (V3 protocol).")
    p.add_argument("--config", default="config/calibration.json")
    p.add_argument("--transport", choices=("tcp", "websocket", "http"))
    p.add_argument("--tcp-host")
    p.add_argument("--tcp-port", type=int)
    p.add_argument("--ws-host")
    p.add_argument("--ws-port", type=int)
    p.add_argument("--camera", type=int)
    p.add_argument("--control-mode", choices=("direct_joint", "ik"))
    p.add_argument(
        "--no-transport", "--no-serial", dest="no_transport", action="store_true",
        help="Run vision/UI without sending commands.",
    )
    p.add_argument("--start-active", action="store_true", help="(kept for compatibility)")
    p.add_argument("--profile", action="store_true", help="Print per-stage frame timings.")
    p.add_argument("--motion-log", metavar="FILE",
                   help="Write a ~15 Hz CSV of pitch target vs profiled angle "
                        "(t, p1_t, p1, p2_t, p2, p3_t, p3, phase) for smoothness analysis.")
    p.add_argument(
        "--no-threaded-capture", dest="no_threaded_capture", action="store_true",
        help="Disable the background capture thread (use synchronous reads).",
    )
    # ---- PS3 / gamepad joystick options -----------------------------------
    p.add_argument(
        "--joystick-index", type=int, default=0, metavar="N",
        help="Pygame joystick index to use (default 0).  Ignored if no joystick "
             "is connected; the system falls back to gesture mode automatically.",
    )
    p.add_argument(
        "--no-joystick", dest="no_joystick", action="store_true",
        help="Disable joystick probing and always use gesture/hand mode.",
    )
    return p.parse_args()


def _apply_runtime_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.transport:
        config.transport.kind = args.transport
        config.transport.enabled = True
    if args.tcp_host:
        config.tcp.host = args.tcp_host
    if args.tcp_port is not None:
        config.tcp.port = args.tcp_port
    if args.ws_host:
        config.websocket.host = args.ws_host
    if args.ws_port is not None:
        config.websocket.port = args.ws_port
    if args.camera is not None:
        config.camera.device_index = args.camera
    if args.control_mode:
        config.mapping.control_mode = args.control_mode
    if args.no_transport:
        config.transport.enabled = False
    if getattr(args, "no_threaded_capture", False):
        config.camera.threaded_capture = False

    kind = config.transport.kind
    config.tcp.enabled = config.transport.enabled and kind == "tcp"
    config.websocket.enabled = config.transport.enabled and kind == "websocket"


_BACKEND_NAMES = {cv2.CAP_DSHOW: "dshow", cv2.CAP_V4L2: "v4l2",
                  cv2.CAP_MSMF: "msmf", cv2.CAP_ANY: "any"}


def _fourcc_str(value: int) -> str:
    value = int(value)
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip() or "----"


def _open_camera(config: AppConfig) -> cv2.VideoCapture:
    index = config.camera.device_index
    system = platform.system().lower()
    if index >= 1:
        config.camera.frame_width = 1280
        config.camera.frame_height = 720
        config.camera.fps = 30

    if system == "windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif system == "linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    indexes = [index] + ([2] if system == "linux" and index == 1 else [])
    cap, used_backend = None, cv2.CAP_ANY
    for try_idx in indexes:
        for backend in backends:
            candidate = cv2.VideoCapture(try_idx, backend)
            if candidate.isOpened():
                cap, used_backend, index = candidate, backend, try_idx
                break
            candidate.release()
        if cap is not None:
            break
    if cap is None:
        raise RuntimeError(f"Could not open camera index {config.camera.device_index}. Try --camera 0 or 2.")

    # Request MJPG BEFORE size/fps: most USB webcams only deliver 720p@30 as
    # MJPG (raw YUYV is bandwidth-limited and often falls back to a tiny mode).
    if config.camera.use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.frame_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"[camera] index={index} backend={_BACKEND_NAMES.get(used_backend, used_backend)} "
          f"mode={aw}x{ah}@{afps:.0f} fourcc={fourcc} "
          f"mjpg={'on' if config.camera.use_mjpg else 'off'} "
          f"threaded={'on' if config.camera.threaded_capture else 'off'}", flush=True)
    if config.camera.use_mjpg and fourcc not in ("MJPG", "MJPEG"):
        print("[camera] note: MJPG not granted; if the feed is small/slow try a "
              "different --camera index or set camera.use_mjpg=false.", flush=True)
    return cap


class CameraStream:
    """Threaded capture that always hands the main loop the NEWEST frame.

    The grab thread keeps draining the driver's frame queue at camera rate; the
    main loop reads whatever is latest, so slow per-frame work never accumulates
    the V4L2 buffer lag (the classic Linux 'stale frame' delay). cv2 returns a
    fresh ndarray per read, so handing out the reference needs no copy."""

    def __init__(self, cap: cv2.VideoCapture, mirror: bool) -> None:
        self._cap = cap
        self._mirror = mirror
        self._frame = None
        self._seq = 0
        self._last_read_seq = -1
        self._err: str | None = None
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                with self._cond:
                    self._err = "Camera frame grab failed."
                    self._cond.notify_all()
                return
            if self._mirror:
                frame = cv2.flip(frame, 1)
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 2.0):
        """Block until a frame newer than the last returned one is ready; drops
        any intermediate frames so the loop always gets the freshest one."""
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._err is not None
                or (self._frame is not None and self._seq != self._last_read_seq),
                timeout=timeout,
            )
            if self._err is not None and self._frame is None:
                raise RuntimeError(self._err)
            if not ready or self._frame is None:
                return None
            self._last_read_seq = self._seq
            return self._frame

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:
            pass


class _SyncCamera:
    """Synchronous fallback with the same interface as CameraStream."""

    def __init__(self, cap: cv2.VideoCapture, mirror: bool) -> None:
        self._cap = cap
        self._mirror = mirror

    def read(self, timeout: float = 2.0):
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Camera frame grab failed.")
        return cv2.flip(frame, 1) if self._mirror else frame

    def stop(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


def _make_camera(config: AppConfig):
    cap = _open_camera(config)
    if config.camera.threaded_capture:
        return CameraStream(cap, config.camera.mirror_view)
    return _SyncCamera(cap, config.camera.mirror_view)


class _StageProfiler:
    """Near-zero-cost per-stage timing; prints mean ms every ~3 s when active."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._acc: dict[str, float] = {}
        self._n = 0
        self._last = time.perf_counter()
        self._t = 0.0

    def start(self) -> None:
        if self.enabled:
            self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._acc[name] = self._acc.get(name, 0.0) + (now - self._t) * 1000.0
        self._t = now

    def tick(self) -> None:
        if not self.enabled:
            return
        self._n += 1
        now = time.perf_counter()
        if now - self._last >= 3.0 and self._n:
            parts = "  ".join(f"{k} {v / self._n:.1f}ms" for k, v in self._acc.items())
            print(f"[profile] {self._n / (now - self._last):.0f} fps | {parts}", flush=True)
            self._acc.clear()
            self._n = 0
            self._last = now


def _pose_from_cfg(pose_cfg) -> Pose:
    return Pose(pose_cfg.base_deg, pose_cfg.p1_deg, pose_cfg.p2_deg,
                pose_cfg.p3_deg, pose_cfg.roll_deg, pose_cfg.gripper_open)


def _active_pose(config: AppConfig) -> Pose:
    return _pose_from_cfg(config.active_pose)


def _home_pose(config: AppConfig) -> Pose:
    return _pose_from_cfg(config.home_pose)


def _make_pitch_limiter(config: AppConfig, joint_cfg, seed: float):
    """Build the configured smoothing profile for one pitch joint.
    Returns None for profile 'none' (host passes raw targets; firmware smooths)."""
    profile = config.motion.pitch_profile
    if profile == "none":
        return None
    if profile == "exp_smooth":
        return ExpSmoothLimiter(joint_cfg.rate_deg_s, config.motion.exp_track_gain,
                                joint_cfg.accel_deg_s2 * 4.0, seed)
    if profile == "trapezoid":
        return AccelSlewLimiter(joint_cfg.rate_deg_s, joint_cfg.accel_deg_s2, seed)
    # default: jerk-limited s-curve with target-velocity feedforward
    return SCurveLimiter(joint_cfg.rate_deg_s, joint_cfg.accel_deg_s2,
                         joint_cfg.jerk_deg_s3, seed)


def _build_filters(config: AppConfig) -> dict:
    a = config.active_pose
    fcfg = config.filters
    m = config.motion
    return {
        "x": make_input_filter(config, fcfg.x_alpha),
        "height": make_input_filter(config, fcfg.y_alpha),
        # depth is the noisiest input -> its own harder One-Euro smoothing.
        "depth": make_input_filter(config, fcfg.depth_alpha,
                                   min_cutoff=fcfg.depth_min_cutoff, beta=fcfg.depth_beta),
        "grip": make_input_filter(config, fcfg.grip_alpha),
        "roll": make_input_filter(config, fcfg.x_alpha),
        # Pitch joints: per-joint selectable profile (see MotionConfig).
        "p1": _make_pitch_limiter(config, m.p1, a.p1_deg),
        "p2": _make_pitch_limiter(config, m.p2, a.p2_deg),
        "p3": _make_pitch_limiter(config, m.p3, a.p3_deg),
        "gripper": SlewRateLimiter(fcfg.gripper_rate_per_s, a.gripper_open),
    }


def _seed_pitch_slew(filters: dict, active: Pose) -> None:
    for key, val in (("p1", active.p1), ("p2", active.p2), ("p3", active.p3)):
        if filters[key] is not None:
            filters[key].reset(val)
    filters["gripper"].reset(active.grip)


def _pitch_bounds(config: AppConfig) -> dict:
    """Output (servo-space) limits per pitch joint. p2 follows the configured
    host direction, so its servo range may be the joint range or its mirror."""
    lim = config.limits
    if config.mapping.p2_invert:
        p2 = (180.0 - lim.p2_max, 180.0 - lim.p2_min)
    else:
        p2 = (lim.p2_min, lim.p2_max)
    return {"p1": (lim.p1_min, lim.p1_max), "p2": p2, "p3": (lim.p3_min, lim.p3_max)}


def _clamp_limiter(limiter, value: float, lo: float, hi: float) -> float:
    """Clamp a smoothing-limiter's output to [lo,hi] with anti-windup: if the
    profile overshoots a limit (s-curve momentum on a fast target), pin its
    internal state at the limit and kill velocity so it can't drive further
    out of range (this is what let p1 dip negative on a quick move)."""
    if value < lo or value > hi:
        value = clamp(value, lo, hi)
        limiter.value = value
        limiter.velocity = 0.0
        if hasattr(limiter, "_v_cmd"):
            limiter._v_cmd = 0.0
        if hasattr(limiter, "accel"):
            limiter.accel = 0.0
    return value


def _slew_scale(target: float, neutral: float, config: AppConfig) -> float:
    """Dynamic slew: slow (fine) near the active pose, gradually faster outward."""
    dist = abs(target - neutral)
    t = clamp(dist / max(config.filters.slew_dist_max_deg, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return config.filters.slew_min_scale + (1.0 - config.filters.slew_min_scale) * smooth


def _ramp_pose_toward(pose: Pose, target: Pose, rate_deg_s: float,
                      grip_rate_frac_s: float, dt: float) -> bool:
    """Advance `pose` linearly toward `target` at rate_deg_s (gripper at the
    equivalent fraction rate). Returns True once every channel has arrived."""
    step = max(0.0, rate_deg_s) * dt
    arrived = True
    for attr in ("base", "p1", "p2", "p3", "roll"):
        cur = getattr(pose, attr)
        tgt = getattr(target, attr)
        delta = clamp(tgt - cur, -step, step)
        setattr(pose, attr, cur + delta)
        if abs(tgt - (cur + delta)) > 0.5:
            arrived = False
    gstep = max(0.0, grip_rate_frac_s) * dt
    gdelta = clamp(target.grip - pose.grip, -gstep, gstep)
    pose.grip += gdelta
    if abs(target.grip - pose.grip) > 0.02:
        arrived = False
    return arrived


class PoseTransition:
    """Minimum-jerk pose-to-pose trajectory with COORDINATED arrival: one
    duration is derived from the largest joint delta, and every joint follows
    the same s(tau) shape, so all joints start, cruise, and stop together
    (instead of finishing one by one and fighting each other).

    Time advances by the dt passed to step(), not wall clock, so freeze/e-stop
    pauses the trajectory instead of skipping ahead on resume."""

    _JOINTS = ("base", "p1", "p2", "p3", "roll")

    def __init__(self, start: Pose, target: Pose, rate_deg_s: float, grip_span_deg: float) -> None:
        self.start = Pose(start.base, start.p1, start.p2, start.p3, start.roll, start.grip)
        self.target = target
        deltas = [abs(getattr(target, j) - getattr(start, j)) for j in self._JOINTS]
        deltas.append(abs(target.grip - start.grip) * max(grip_span_deg, 1e-6))
        # Min-jerk peak velocity is 1.875x the average; size the duration so the
        # PEAK equals the configured rate (rate keeps its "deg/s" meaning).
        self.duration = max(1.875 * max(deltas) / max(rate_deg_s, 0.1), 0.4)
        self.t = 0.0

    def step(self, pose: Pose, dt: float) -> bool:
        self.t += max(0.0, dt)
        s = min_jerk_s(self.t / self.duration)
        for joint in self._JOINTS:
            a = getattr(self.start, joint)
            b = getattr(self.target, joint)
            setattr(pose, joint, a + (b - a) * s)
        pose.grip = self.start.grip + (self.target.grip - self.start.grip) * s
        if self.t >= self.duration:
            pose.set_from(self.target)
            return True
        return False


def _drive_from_hand(signals, dt: float, now: float, config: AppConfig,
                     filters: dict, pose: Pose, cs: ControlState,
                     live_scale: float = 1.0) -> None:
    """Active-control update for one frame: dwell-gated holds -> filter ->
    integrate -> accel-limited slew. Mutates `pose` and `cs` in place."""
    fcfg = config.filters
    curl = signals.finger_curl_norm
    tilt = abs(signals.roll_tilt_delta)

    def _filter_value(name: str, fallback: float) -> float:
        v = filters[name].value
        return v if v is not None else fallback

    # --- dwell trackers with pre-gesture snapshots -------------------------
    # Snapshot depth/height the moment the trigger condition first appears, so
    # that if the hold confirms after the dwell, we hold the PRE-gesture value
    # (curling fingers / tilting the hand perturbs the depth proxy meanwhile).
    if tilt > fcfg.roll_active_tilt_threshold:
        if cs.roll_above_since is None:
            cs.roll_above_since = now
            cs.roll_depth_snapshot = _filter_value("depth", signals.depth_norm)
            cs.roll_height_snapshot = _filter_value("height", signals.height_norm)
    else:
        cs.roll_above_since = None
    cs.roll_active = (
        cs.roll_above_since is not None
        and (now - cs.roll_above_since) >= fcfg.hold_engage_dwell_s
    )

    if curl >= fcfg.depth_hold_curl_threshold:
        if cs.grip_above_since is None:
            cs.grip_above_since = now
            cs.grip_depth_snapshot = _filter_value("depth", signals.depth_norm)
    else:
        cs.grip_above_since = None

    # Grip-triggered depth hold: pause depth while grabbing, then resume after
    # a few seconds even if the fist stays closed (so a grabbed object can be
    # moved). Re-arms only after the hand opens clearly.
    if curl <= fcfg.depth_hold_release_curl_threshold:
        cs.grip_hold_ready = True
    if cs.grip_hold_active:
        if (now - cs.grip_hold_start) >= fcfg.grip_depth_hold_s:
            cs.grip_hold_active = False
        elif curl <= fcfg.depth_hold_release_curl_threshold:
            cs.grip_hold_active = False
    grip_engage = (
        cs.grip_above_since is not None
        and (now - cs.grip_above_since) >= fcfg.hold_engage_dwell_s
    )
    if (not cs.grip_hold_active) and cs.grip_hold_ready and grip_engage:
        cs.grip_hold_active = True
        cs.grip_hold_ready = False
        cs.grip_hold_start = now

    cs.depth_hold = cs.grip_hold_active or cs.roll_active  # roll perturbs depth too

    if cs.depth_hold and not cs.depth_hold_prev:
        if cs.grip_hold_active and cs.grip_depth_snapshot is not None:
            cs.held_depth = cs.grip_depth_snapshot
        elif cs.roll_depth_snapshot is not None:
            cs.held_depth = cs.roll_depth_snapshot
        else:
            cs.held_depth = signals.depth_norm
    if cs.roll_active and not cs.roll_active_prev:
        cs.held_height = (
            cs.roll_height_snapshot
            if cs.roll_height_snapshot is not None
            else signals.height_norm
        )

    raw_depth = cs.held_depth if cs.depth_hold else signals.depth_norm
    raw_height = cs.held_height if cs.roll_active else signals.height_norm  # roll perturbs height too

    # Filter inputs (One-Euro or low-pass per config)
    x_f = filters["x"].update(signals.x_offset_norm, dt)
    h_f = filters["height"].update(raw_height, dt)
    d_f = filters["depth"].update(raw_depth, dt)
    roll_f = filters["roll"].update(signals.roll_input_norm, dt)
    g_f = filters["grip"].update(signals.gripper_open, dt)
    if cs.depth_hold:
        filters["depth"].reset(cs.held_depth)
        d_f = cs.held_depth
    if cs.roll_active:
        filters["height"].reset(cs.held_height)
        h_f = cs.held_height

    # Yaw & roll: integrated-rate, deadband + gradual (non-linear) acceleration
    pose.base = compute_base_angle(pose.base, x_f, dt, config, live_scale)
    pose.roll = compute_roll_angle(pose.roll, roll_f, dt, config, live_scale)

    # Pitch joints (direct or IK) through the selected per-joint profile.
    # The dynamic-slew cap modulation only applies to the legacy trapezoid: it
    # varies the speed cap mid-move (reads as accel-slow-accel on long sweeps).
    p1t, p2t, p3t = compute_pitch_targets(h_f, d_f, cs.prev_pitch, config)
    a = config.active_pose
    use_dyn = (config.motion.pitch_profile == "trapezoid"
               and config.motion.use_dynamic_slew)
    bounds = _pitch_bounds(config)
    for key, target, neutral in (("p1", p1t, a.p1_deg),
                                 ("p2", p2t, a.p2_deg),
                                 ("p3", p3t, a.p3_deg)):
        lo, hi = bounds[key]
        limiter = filters[key]
        if limiter is None:  # profile "none": firmware does all the smoothing
            setattr(pose, key, clamp(target, lo, hi))
            continue
        scale = (_slew_scale(target, neutral, config) if use_dyn else 1.0) * live_scale
        out = _clamp_limiter(limiter, limiter.update(target, dt, scale), lo, hi)
        setattr(pose, key, out)
    pose.grip = filters["gripper"].update(g_f, dt, live_scale)

    cs.prev_pitch = (p1t, p2t, p3t)
    cs.depth_hold_prev = cs.depth_hold
    cs.roll_active_prev = cs.roll_active
    cs.reach_r, cs.reach_z = reach_height_to_rz(d_f, h_f, config)


def _drive_two_hand(th, dt: float, config: AppConfig, filters: dict, pose: Pose,
                    live_scale: float = 1.0) -> None:
    """Apply one TwoHandFrame to the pose through the existing smoothing stack.
    None targets hold; yaw is rate-integrated from the combined (per-hand
    deadbanded) deflection; everything is clamped to the joint bounds."""
    lim = config.limits
    pose.base = integrate_rate(pose.base, th.yaw_input, 0.0,
                               config.mapping.yaw_exponent,
                               config.mapping.yaw_rate_deg_s * max(0.0, live_scale),
                               dt, lim.base_min, lim.base_max)
    pose.roll = compute_roll_angle(pose.roll, th.roll_input, dt, config, live_scale)

    bounds = _pitch_bounds(config)
    for key, tgt in (("p1", th.p1), ("p2", th.p2), ("p3", th.p3)):
        lo, hi = bounds[key]
        limiter = filters[key]
        if tgt is None:
            # HOLD (locked/paused or hand absent): freeze the limiter AT the
            # current value. Feeding it its own output as the target lets any
            # residual velocity chase itself and drift to a limit — reset()
            # pins value and zeroes velocity so p1/p2 stay put.
            if limiter is not None:
                limiter.reset(getattr(pose, key))
            continue
        if limiter is None:
            setattr(pose, key, clamp(tgt, lo, hi))
            continue
        out = _clamp_limiter(limiter, limiter.update(tgt, dt, live_scale), lo, hi)
        setattr(pose, key, out)

    if th.grip is not None:
        pose.grip = filters["gripper"].update(th.grip, dt, live_scale)


class TransportPump:
    """Streams the LATEST command to the transport on its own thread at a
    fixed rate, decoupled from the camera loop. Two failure modes die here:
    a MediaPipe/camera stall no longer starves the firmware watchdog (the
    pump keeps re-sending the last pose as a keepalive), and a socket op can
    never block the vision loop (all sends happen on this thread)."""

    def __init__(self, transport, hz: float) -> None:
        self._transport = transport
        self._interval = 1.0 / max(hz, 1.0)
        self._command = None          # latest TeleopCommand; ref swap is atomic
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tx_pump")
        self._thread.start()

    def update(self, command) -> None:
        self._command = command

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            cmd = self._command
            if cmd is not None:
                try:
                    # the pump IS the pacer, so bypass the per-send throttle
                    self._transport.send(cmd, force=True)
                except Exception:
                    pass  # transport reports via last_error; never kill the pump
            self._stop_evt.wait(self._interval)

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=1.0)


def _transport_status(config: AppConfig, transport) -> str:
    if not config.transport.enabled:
        return "disabled"
    if transport.connected:
        if config.transport.kind == "tcp":
            return f"tcp {config.tcp.host}:{config.tcp.port}"
        return f"websocket {config.websocket.host}:{config.websocket.port}"
    return f"offline: {transport.last_error}" if transport.last_error else "offline"


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    _apply_runtime_overrides(config, args)

    # ------------------------------------------------------------------
    # PS3 joystick probe: try first; fall back to gesture mode silently.
    # ------------------------------------------------------------------
    joystick: JoystickDriver | None = None
    if not getattr(args, "no_joystick", False):
        joystick = probe_joystick(
            index=getattr(args, "joystick_index", 0),
            verbose=True,
            config=config,
        )
    use_joystick = joystick is not None
    if use_joystick:
        print("[main] Control mode: PS3 JOYSTICK  (camera and MediaPipe DISABLED)")
        # Re-seed joystick with config-derived rates so it matches gesture feel
        joystick.set_rates_from_config(config)
    else:
        print("[main] Control mode: GESTURE / HAND  (no joystick detected)")

    # Only initialise camera-dependent objects when gesture mode is active.
    tracker  = HandTracker(config.vision) if not use_joystick else None
    two_hand = TwoHandController(config)  if not use_joystick else None
    use_two_hand = config.control.scheme == "two_hand" and not use_joystick
    transport = build_transport(config)
    filters = _build_filters(config)
    # TeleopUI (OpenCV window) only in gesture mode
    ui = TeleopUI(WINDOW_NAME) if not use_joystick else None
    profiler = _StageProfiler(args.profile)
    cam = None
    pump_hz = config.tcp.write_hz if config.transport.kind == "tcp" else config.websocket.write_hz
    pump = TransportPump(transport, pump_hz)

    motion_log = None
    motion_log_last = 0.0
    if args.motion_log:
        motion_log = open(args.motion_log, "w", encoding="ascii", buffering=1)
        motion_log.write("t,p1_target,p1,p2_target,p2,p3_target,p3,phase\n")

    active = _active_pose(config)
    home = _home_pose(config)

    # Boot at the home pose: the firmware parks itself there on power-up and
    # the host confirms with mode 'M' until a hand shows up.
    pose = _pose_from_cfg(config.home_pose)
    cs = ControlState(prev_pitch=(active.p1, active.p2, active.p3))

    # Seed the joystick driver at the home pose so there is no jump
    # when the first frame is processed.
    if joystick is not None:
        joystick.reseed(
            base=pose.base, p1=pose.p1, p2=pose.p2,
            p3=pose.p3,     roll=pose.roll, grip=pose.grip,
        )

    # Phases: home -> to_active -> control -> lost_hold -> to_home -> home
    phase = "home"
    hand_lost_since: float | None = None
    frozen = False
    estop = False
    fps_smoothed = 0.0

    transition: PoseTransition | None = None
    transition_phase = ""
    rate = config.control.pose_transition_rate_deg_s
    grip_ramp_rate = dt = 0.0
    # abs(): open/closed servo endpoints may be in either order (open=166.5 > closed=36)
    grip_span = max(abs(config.gripper.closed_deg - config.gripper.open_deg), 1e-6)

    def reseed_from(signals) -> None:
        filters["x"].reset(signals.x_offset_norm)
        filters["height"].reset(signals.height_norm)
        filters["depth"].reset(signals.depth_norm)
        filters["roll"].reset(signals.roll_input_norm)
        filters["grip"].reset(signals.gripper_open)

    def advance_ramp(target: Pose, phase_name: str) -> bool:
        """Advance the pose toward `target` using the configured transition
        profile. min_jerk creates one coordinated trajectory per phase entry."""
        nonlocal transition, transition_phase
        if config.control.transition_profile == "linear":
            return _ramp_pose_toward(pose, target, rate, grip_ramp_rate, dt)
        if transition is None or transition_phase != phase_name:
            transition = PoseTransition(pose, target, rate, abs(grip_span))
            transition_phase = phase_name
        done = transition.step(pose, dt)
        if done:
            transition = None
            transition_phase = ""
        return done

    try:
        if not use_joystick:
            # ---- gesture mode: open camera + warm up MediaPipe ----
            cam = _make_camera(config)
            warm = np.zeros((config.camera.frame_height, config.camera.frame_width, 3), np.uint8)
            for _ in range(2):
                tracker.process(warm)

        previous_time = time.perf_counter()

        while True:
            profiler.start()

            # ================================================================
            # JOYSTICK MODE: no camera, no MediaPipe
            # ================================================================
            if use_joystick:
                now = time.perf_counter()
                dt = min(max(1e-3, now - previous_time), 0.1)
                previous_time = now
                fps_smoothed += 0.1 * ((1.0 / dt) - fps_smoothed)

                js_state: JoystickState | None = None
                try:
                    js_state = joystick.read(dt)
                    if js_state.estop:
                        estop = not estop
                    if js_state.freeze:
                        frozen = not frozen
                    if js_state.go_home and not estop and not frozen:
                        phase = "to_home"
                        hand_lost_since = None
                        cs.reset_locks()
                    if js_state.reset and not estop and not frozen:
                        joystick.reseed(
                            base=home.base, p1=home.p1, p2=home.p2,
                            p3=home.p3,     roll=home.roll, grip=home.grip,
                        )
                        _seed_pitch_slew(filters, home)
                except Exception as js_exc:
                    print(f"[joystick] read error: {js_exc}", flush=True)
                    js_state = None

                frame = None
                hand = None
                hands = []
                th_frame = None
                signals = None
                tracking_ok = False
                hand_active = js_state is not None
                profiler.mark("capture")
                profiler.mark("mediapipe")

                # Throttle to ~50 Hz in joystick mode
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - now)))

            # ================================================================
            # GESTURE MODE: read frame + run MediaPipe
            # ================================================================
            else:
                frame = cam.read()
                if frame is None:
                    if (cv2.waitKeyEx(1) & 0xFF) in (ord("q"), ord("Q")):
                        break
                    continue
                profiler.mark("capture")

                now = time.perf_counter()
                dt = min(max(1e-3, now - previous_time), 0.1)
                previous_time = now
                fps_smoothed += 0.1 * ((1.0 / dt) - fps_smoothed)

                js_state = None
                hands = tracker.process(frame)
                profiler.mark("mediapipe")
                tracking_ok = bool(hands)
                if use_two_hand:
                    th_frame = two_hand.update(hands, pose, now, config)
                    hand = None
                    signals = None
                    hand_active = th_frame.right_present or th_frame.left_present
                else:
                    th_frame = None
                    hand = best_of(hands)
                    signals = extract_signals(hand, config) if hand is not None else None
                    hand_active = bool(signals and signals.inside_region)

            rate = config.control.pose_transition_rate_deg_s
            live = config.control.live_speed_scale
            grip_span = max(abs(config.gripper.closed_deg - config.gripper.open_deg), 1e-6)
            grip_ramp_rate = rate / grip_span

            # ---------------- state machine ----------------
            if estop:
                mode, status = "H", "E-STOP hold"
            elif frozen:
                mode, status = "H", "frozen (hold)"
            elif phase == "home":
                pose.set_from(home)
                mode, status = "M", "home (waiting for hand)"
                if hand_active:
                    phase = "to_active"
            elif phase == "to_active":
                arrived = advance_ramp(active, "to_active")
                mode, status = "A", f"moving to active pose ({rate:.0f} deg/s)"
                if arrived:
                    if hand_active:
                        phase = "control"
                        cs.reset_locks()
                        cs.prev_pitch = (active.p1, active.p2, active.p3)
                        _seed_pitch_slew(filters, active)
                        # Seed joystick at the arrived active pose so
                        # there is no discontinuity at first control frame.
                        if use_joystick and joystick is not None:
                            joystick.reseed(
                                base=pose.base, p1=pose.p1, p2=pose.p2,
                                p3=pose.p3,     roll=pose.roll, grip=pose.grip,
                            )
                        if signals is not None:
                            reseed_from(signals)
                    else:
                        phase = "lost_hold"
                        hand_lost_since = now
            elif phase == "control":
                if hand_active:
                    hand_lost_since = None
                    if use_joystick and js_state is not None:
                        # ---- PS3 joystick mode: direct joint targets ----
                        # Only update when not transitioning so we don't
                        # fight the ramp trajectory.
                        bounds = _pitch_bounds(config)
                        lim = config.limits
                        pose.base = clamp(js_state.base_deg, lim.base_min, lim.base_max)
                        
                        p1_tgt = clamp(js_state.p1_deg, bounds["p1"][0], bounds["p1"][1])
                        p3_tgt = clamp(js_state.p3_deg, bounds["p3"][0], bounds["p3"][1])
                        
                        # Apply the configured p2 direction just like in gesture mode.
                        p2_target = 180.0 - js_state.p2_deg if config.mapping.p2_invert else js_state.p2_deg
                        p2_tgt = clamp(p2_target, bounds["p2"][0], bounds["p2"][1])
                        
                        # Apply host-side filters to avoid joint lag / drift
                        if filters["p1"] is not None:
                            pose.p1 = _clamp_limiter(filters["p1"], filters["p1"].update(p1_tgt, dt, live), bounds["p1"][0], bounds["p1"][1])
                        else:
                            pose.p1 = p1_tgt
                            
                        if filters["p2"] is not None:
                            pose.p2 = _clamp_limiter(filters["p2"], filters["p2"].update(p2_tgt, dt, live), bounds["p2"][0], bounds["p2"][1])
                        else:
                            pose.p2 = p2_tgt
                            
                        if filters["p3"] is not None:
                            pose.p3 = _clamp_limiter(filters["p3"], filters["p3"].update(p3_tgt, dt, live), bounds["p3"][0], bounds["p3"][1])
                        else:
                            pose.p3 = p3_tgt
                            
                        pose.roll = clamp(js_state.roll_deg,  lim.roll_min, lim.roll_max)
                        pose.grip = filters["gripper"].update(js_state.grip_open, dt, live)
                        status = f"joystick active  roll={pose.roll:.1f}°"
                    elif use_two_hand:
                        _drive_two_hand(th_frame, dt, config, filters, pose, live)
                        r_state = ("LOCKED" if th_frame.right_locked
                                   else "active" if th_frame.right_engaged else "--")
                        l_state = "active" if th_frame.left_engaged else "--"
                        status = f"two-hand   R {r_state}   L {l_state}"
                    else:
                        _drive_from_hand(signals, dt, now, config, filters, pose, cs, live)
                        status = "teleop (grasp-hold)" if cs.depth_hold else "teleop active"
                    mode = "A"
                else:
                    phase = "lost_hold"
                    hand_lost_since = now
                    cs.reset_locks()
                    mode, status = "H", "tracking lost, holding"
            elif phase == "lost_hold":
                mode = "H"
                elapsed = now - (hand_lost_since if hand_lost_since is not None else now)
                if hand_active:
                    phase = "control"
                    hand_lost_since = None
                    cs.reset_locks()
                    if signals is not None:
                        reseed_from(signals)
                    status = "teleop active"
                elif elapsed > config.control.lost_freeze_s:
                    phase = "to_home"
                    status = "going home"
                else:
                    remaining = config.control.lost_freeze_s - elapsed
                    status = f"tracking lost, holding {remaining:.0f}s"
            else:  # to_home
                arrived = advance_ramp(home, "to_home")
                mode, status = "A", f"going home ({rate:.0f} deg/s)"
                if arrived:
                    phase = "home"

            # thumbs-up gesture (either hand, dwelled) -> go home
            if (use_two_hand and th_frame is not None and th_frame.home_requested
                    and phase in ("control", "lost_hold") and not estop and not frozen):
                phase = "to_home"
                hand_lost_since = None
                cs.reset_locks()

            # drop any stale trajectory once we're out of the ramp phases
            if phase not in ("to_active", "to_home"):
                transition = None
                transition_phase = ""

            command = pose.command(mode, config.gripper)
            pump.update(command)  # the pump thread streams it at write_hz
            transport_status = _transport_status(config, transport)

            if motion_log is not None and (now - motion_log_last) >= (1.0 / 15.0):
                motion_log_last = now
                t1, t2, t3 = cs.prev_pitch if phase == "control" else (pose.p1, pose.p2, pose.p3)
                motion_log.write(f"{now:.3f},{t1:.2f},{pose.p1:.2f},{t2:.2f},{pose.p2:.2f},"
                                 f"{t3:.2f},{pose.p3:.2f},{phase}\n")
            profiler.mark("control")

            # ----------------------------------------------------------------
            # OUTPUT: joystick → rolling log line; gesture → OpenCV overlay
            # ----------------------------------------------------------------
            if use_joystick:
                # Single overwriting line so the terminal stays clean
                estop_tag  = "  [E-STOP]" if estop  else ""
                frozen_tag = "  [FROZEN]" if frozen else ""
                print(
                    f"\r[JS] {status:<28} | "
                    f"base={pose.base:6.1f}  p1={pose.p1:5.1f}  p2={pose.p2:5.1f}  "
                    f"p3={pose.p3:5.1f}  roll={pose.roll:5.1f}  grip={pose.grip:.2f}  "
                    f"tx={transport_status}{estop_tag}{frozen_tag}",
                    end="", flush=True,
                )
                profiler.mark("overlay")
            else:
                # ---- build UI data ----
                if use_two_hand and th_frame is not None:
                    hands2 = [
                        {"obs": th_frame.right_obs, "role": "R",
                         "engaged": th_frame.right_engaged, "locked": th_frame.right_locked,
                         "anchor": th_frame.right_anchor, "deflection": th_frame.right_deflection},
                        {"obs": th_frame.left_obs, "role": "L",
                         "engaged": th_frame.left_engaged, "locked": False,
                         "anchor": th_frame.left_anchor, "deflection": th_frame.left_deflection},
                    ]
                    pb = _pitch_bounds(config)
                    ui_x = 0.5 * (th_frame.yaw_input + 1.0)
                    ui_h = (pose.p2 - pb["p2"][0]) / max(pb["p2"][1] - pb["p2"][0], 1e-6)
                    ui_d = (pose.p1 - pb["p1"][0]) / max(pb["p1"][1] - pb["p1"][0], 1e-6)
                    ui_g = th_frame.grip if th_frame.grip is not None else pose.grip
                    ui_r = th_frame.roll_input
                    ui_rh, ui_lh = th_frame.r_height, th_frame.l_height
                else:
                    hands2 = None
                    ui_x = 0.5 * ((signals.x_offset_norm if signals else 0.0) + 1.0)
                    ui_h = signals.height_norm if signals else 0.5
                    ui_d = signals.depth_norm if signals else 0.5
                    ui_g = signals.gripper_open if signals else pose.grip
                    ui_r = signals.roll_input_norm if signals else 0.0
                    ui_rh = ui_lh = None

                ui.render(frame, config, hand, {
                    "status": status,
                    "phase": phase,
                    "mode": mode,
                    "control_mode": config.mapping.control_mode,
                    "scheme": config.control.scheme,
                    "hands2": hands2,
                    "tracking_ok": tracking_ok,
                    "hand_active": hand_active,
                    "frozen": frozen,
                    "estop": estop,
                    "base_deg": pose.base,
                    "p1_deg": pose.p1,
                    "p2_deg": pose.p2,
                    "p3_deg": pose.p3,
                    "roll_deg": pose.roll,
                    "gripper_open": pose.grip,
                    "gripper_deg": command.gripper_deg,
                    "depth_hold": cs.depth_hold if not use_two_hand else False,
                    "roll_active": cs.roll_active if not use_two_hand else False,
                    "reach_r": cs.reach_r,
                    "reach_z": cs.reach_z,
                    "transport_status": transport_status,
                    "transition_rate": rate,
                    "live_scale": live,
                    "fps": fps_smoothed,
                    "x_norm": ui_x,
                    "height_norm": ui_h,
                    "r_height": ui_rh,
                    "l_height": ui_lh,
                    "depth_norm": ui_d,
                    "grip_norm": ui_g,
                    "roll_input": ui_r,
                })
                profiler.mark("overlay")

            # Key handling: gesture mode only (joystick buttons handle control)
            if not use_joystick:
                key = cv2.waitKeyEx(1)
                profiler.mark("wait")
                profiler.tick()
                if key == -1:
                    continue
                low = key & 0xFF

                if low in (ord("q"), ord("Q")):
                    break
                elif low in (ord("f"), ord("F")):
                    frozen = not frozen
                    if not frozen and signals is not None:
                        reseed_from(signals)
                elif low in (ord("x"), ord("X")):
                    estop = not estop
                elif low in (ord("h"), ord("H")):
                    phase = "to_home"
                    hand_lost_since = None
                    cs.reset_locks()
                elif low in (ord("n"), ord("N")):
                    if use_two_hand:
                        if hands:
                            l_o, r_o = two_hand.assign_roles(hands, config)
                            two_hand.set_anchors(l_o, r_o, pose, config)
                            if r_o is not None:
                                config.vision.neutral_right_x, config.vision.neutral_right_y = r_o.center_xy
                            if l_o is not None:
                                config.vision.neutral_left_x, config.vision.neutral_left_y = l_o.center_xy
                                config.vision.grip_open_reference = l_o.finger_curl_metric
                                config.vision.grip_closed_reference = (
                                    l_o.finger_curl_metric + config.vision.grip_close_span)
                            save_config(config)
                    elif hand is not None:
                        capture_neutral_reference(config, hand)
                        save_config(config)
                        reseed_from(extract_signals(hand, config))
                elif key in KEY_UP:
                    config.control.pose_transition_rate_deg_s = clamp(rate + 1.0, 1.0, 30.0)
                    save_config(config)
                elif key in KEY_DOWN:
                    config.control.pose_transition_rate_deg_s = clamp(rate - 1.0, 1.0, 30.0)
                    save_config(config)
                elif low in (ord("="), ord("+")):
                    config.control.live_speed_scale = clamp(live + 0.25, 0.25, 2.0)
                    save_config(config)
                elif low in (ord("-"), ord("_")):
                    config.control.live_speed_scale = clamp(live - 0.25, 0.25, 2.0)
                    save_config(config)
            else:
                profiler.mark("wait")
                profiler.tick()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
        pump.stop()
        try:
            transport.send(pose.command("H", config.gripper), force=True)
        except Exception:
            pass
        transport.close()
        if tracker is not None:
            tracker.close()
        if cam is not None:
            cam.stop()
        if motion_log is not None:
            motion_log.close()
        shutdown_joystick()
        if not use_joystick:
            cv2.destroyAllWindows()
        else:
            print()  # newline after the rolling log line

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
