from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T")


@dataclass
class RegionConfig:
    x_min: float = 0.08
    y_min: float = 0.08
    x_max: float = 0.92
    y_max: float = 0.92

    @property
    def width(self) -> float:
        return max(1e-6, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(1e-6, self.y_max - self.y_min)


@dataclass
class CameraConfig:
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 60
    mirror_view: bool = True
    # Request MJPG from the camera: enables 720p@30 with low latency on most
    # USB webcams (raw YUYV is bandwidth-limited and often forces a tiny/slow mode).
    use_mjpg: bool = True
    # Grab frames on a background thread and always process the NEWEST one.
    # Removes the V4L2 "stale frame" buffer lag on Linux.
    threaded_capture: bool = True
    # False = letterbox (preserve whole frame); True = crop to fill the window.
    fill_window: bool = False


@dataclass
class VisionConfig:
    # 2 for the two-hand scheme (left/right roles); direct mode uses the best one.
    max_num_hands: int = 2
    # MediaPipe Hands complexity: 0 = fast/low-latency, 1 = accurate.
    model_complexity: int = 1
    min_detection_confidence: float = 0.45
    min_tracking_confidence: float = 0.35
    handedness: str = "either"
    active_region: RegionConfig = field(default_factory=RegionConfig)
    neutral_x: float = 0.5
    neutral_y: float = 0.55
    depth_reference: float = 0.18
    wrist_tilt_reference: float = 0.0
    grip_open_reference: float = 0.2
    grip_closed_reference: float = 0.75
    # Closed reference derived on 'n' capture: closed = open + grip_close_span.
    grip_close_span: float = 0.55
    pinch_open_reference: float = 0.95
    pinch_closed_reference: float = 0.28
    min_palm_width_norm: float = 0.015
    # Initial per-role anchor points for the two-hand scheme (markers shown as
    # the L/R Agnisys logos). Each engagement re-anchors to the live hand, so
    # these only seed the display / first engagement.
    neutral_left_x: float = 0.27
    neutral_left_y: float = 0.55
    neutral_right_x: float = 0.73
    neutral_right_y: float = 0.55


@dataclass
class MappingConfig:
    control_mode: str = "direct_joint"
    yaw_deadband: float = 0.12
    yaw_exponent: float = 1.6
    yaw_max_command: float = 1.0
    yaw_invert: bool = True
    base_trim_command: float = 0.0
    depth_to_reach_sign: float = 1.0
    # Depth sensitivity (per unit of the depth signal). Lower = less sensitive /
    # more precise; raise if you can't reach the angles you need.
    depth_gain: float = 0.3
    # depth_symmetric=True uses log(size_ratio) as the depth signal. Apparent
    # hand size grows without bound as the hand nears the camera but only
    # shrinks to a floor as it recedes, so the plain (ratio-1) signal is
    # LOPSIDED: the "far" side (p1 -> max) is compressed and hard to reach.
    # log() is symmetric (equal reach for equal DISTANCE factor closer/farther),
    # so the full 0-90 p1 range is accessible with normal hand travel and the
    # feel is uniform per distance-factor. False = the old linear (ratio-1) map.
    depth_symmetric: bool = True
    # Optional localized deadband around neutral only. OFF by default (0.0): it
    # rescales past the dead zone (dead at neutral, then STEEPER), so it does
    # NOT give uniform sensitivity. Use depth_gain for that; only enable this if
    # you specifically want a hold zone at the neutral depth.
    depth_deadband: float = 0.0
    lower_depth_invert: bool = True
    lower_height_invert: bool = False
    middle_height_invert: bool = True
    upper_height_invert: bool = False
    middle_average_weight: float = 0.5
    # p2 servo is mounted flipped: command = 180 - avg(p1, p3). With p1=p3=0
    # the p2 servo must sit at 180.
    p2_invert: bool = False
    upper_tilt_sign: float = -1.0
    upper_tilt_gain: float = 1.8
    upper_tilt_deadband: float = 0.06
    gripper_exponent: float = 1.0
    # False (natural): open hand -> gripper OPEN, closed fist -> gripper CLOSED.
    gripper_invert: bool = False
    # Proportional gripper: keep the snap band tiny so openness tracks curl
    # continuously (only the last 5% snaps to fully open/closed).
    gripper_full_open_threshold: float = 0.05
    gripper_full_close_threshold: float = 0.95
    wrist_depth_compensation_gain: float = 0.3
    height_depth_compensation_gain: float = 0.35
    wrist_yaw_compensation_gain: float = 0.25
    # Roll->depth feedforward decoupling: cancel this many degrees of p1 at
    # full roll (rolling changes apparent hand size, which spikes depth->p1).
    # Flip the sign if it makes the lean worse instead of better.
    roll_depth_comp_deg: float = 5.0
    # --- base yaw driven as an integrated rate (deadband + non-linear accel,
    #     holds position until an opposite input is given) ---
    yaw_rate_deg_s: float = 25.0  # base swings the whole arm: slow = smooth
    # --- roll driven as an integrated rate from hand tilt (same feel as yaw) ---
    # 0.29 * 0.61 rad span ~= +-10 deg of tilt before any roll motion starts.
    roll_deadband: float = 0.29
    roll_exponent: float = 1.6
    roll_rate_deg_s: float = 40.0
    roll_tilt_span_rad: float = 0.61  # ~35 deg of hand tilt -> full-speed roll


@dataclass
class WorkspaceConfig:
    target_r_min_mm: float = 70.0
    target_r_max_mm: float = 185.0
    target_z_min_mm: float = 35.0
    target_z_max_mm: float = 185.0
    default_r_mm: float = 105.0
    default_z_mm: float = 155.0
    lost_hold_timeout_s: float = 0.35


@dataclass
class FilterConfig:
    x_alpha: float = 0.45
    y_alpha: float = 0.45
    depth_alpha: float = 0.24
    grip_alpha: float = 0.34
    reach_rate_mm_s: float = 130.0
    height_rate_mm_s: float = 160.0
    joint_rate_deg_s: float = 45.0
    # Pitch joints accelerate gradually from standstill (host-side trapezoid).
    joint_accel_deg_s2: float = 90.0
    gripper_rate_per_s: float = 1.5
    grasp_arm_slowdown_factor: float = 0.2
    grasp_close_deadband: float = 0.02
    # Grasp-triggered depth hold: freeze depth as soon as the fingers begin to
    # curl (10% grip), so grabbing/placing doesn't spike p1. Release well below
    # the trigger for clean hysteresis.
    depth_hold_curl_threshold: float = 0.10
    depth_hold_release_curl_threshold: float = 0.05
    depth_hold_max_s: float = 6.0
    # --- input filter selection ---
    filter_mode: str = "one_euro"          # "lowpass" | "one_euro"
    one_euro_min_cutoff: float = 1.0  # lower = steadier hover, tiny extra lag
    one_euro_beta: float = 0.02
    one_euro_dcutoff: float = 1.0
    # Dedicated (harder) One-Euro smoothing for the DEPTH channel only — the
    # depth proxy (apparent hand size) is the noisiest input. Lower min_cutoff
    # = steadier hold; lower beta = less jitter passthrough on motion.
    depth_min_cutoff: float = 0.6
    depth_beta: float = 0.008
    # --- dynamic slew: slow (fine) near neutral, faster further out ---
    slew_min_scale: float = 0.35
    slew_dist_max_deg: float = 90.0
    # --- decoupling / depth-hold while grabbing or rolling ---
    grip_depth_hold_s: float = 4.5         # pause depth ~4-5 s while grabbing, then resume
    roll_active_tilt_threshold: float = 0.1745  # rad (~10 deg); roll "active" above this
    # A hold only engages after the trigger condition persists this long,
    # so brief flickers of curl/tilt do not cause unintended depth holds.
    hold_engage_dwell_s: float = 0.25


@dataclass
class KinematicsConfig:
    shoulder_r_offset_mm: float = 10.2
    shoulder_z_offset_mm: float = 55.37
    l1_mm: float = 65.92
    l2_mm: float = 66.42
    l3_mm: float = 62.5
    q1_min_deg: float = 0.0
    q1_max_deg: float = 50.0
    q2_min_deg: float = 0.0
    q2_max_deg: float = 60.0
    q3_min_deg: float = 0.0
    q3_max_deg: float = 90.0
    tool_angle_min_deg: float = -90.0
    tool_angle_max_deg: float = 90.0
    tool_angle_step_deg: float = 2.0
    preferred_tool_angle_deg: float = 55.0
    orientation_preference_weight: float = 0.08
    continuity_weight: float = 0.35


@dataclass
class TransportConfig:
    kind: str = "websocket"
    enabled: bool = True


@dataclass
class TcpConfig:
    enabled: bool = True
    host: str = "192.168.1.34"
    port: int = 4210
    timeout_s: float = 5.0
    write_hz: float = 60.0
    dry_run: bool = False
    reconnect_interval_s: float = 2.0  # min gap between non-blocking reconnects


@dataclass
class WebSocketConfig:
    enabled: bool = True
    host: str = "192.168.1.34"
    port: int = 4210
    timeout_s: float = 5.0
    write_hz: float = 30.0
    open_retries: int = 3
    retry_delay_s: float = 0.5
    dry_run: bool = False
    reconnect_interval_s: float = 2.0  # min gap between non-blocking reconnects


@dataclass
class TwoHandConfig:
    """Split-screen two-hand scheme: RIGHT half hand -> yaw + p1(depth) +
    p2(height); LEFT half hand -> yaw + p3(height) + roll(tilt) + grip(curl).
    Right fist LOCKS yaw/p1/p2; reopening re-anchors (no jump)."""
    # right-hand lock (fist) hysteresis + dwells
    lock_curl: float = 0.60
    unlock_curl: float = 0.40
    lock_dwell_s: float = 0.20
    reengage_dwell_s: float = 0.20
    # role assignment: midline hysteresis so an engaged hand keeps its half
    role_hysteresis: float = 0.04
    # channel gains
    yaw_deflection_span: float = 0.25      # x-deflection for full yaw rate
    p2_height_gain_deg: float = 60.0       # p2 servo deg per half-screen of height
    p3_height_gain_deg: float = 90.0       # p3 deg per half-screen of height
    # tracking-dropout grace: a hand absent for less than this keeps its last
    # targets (coast), so one-frame MediaPipe flickers can't stall the motion
    track_grace_s: float = 0.30
    # pose the arm assumes at each hand's neutral (set by pressing N). Right N
    # -> p1/p2 ; left N -> p3 + roll (+ gripper opens with an open left palm).
    neutral_p1: float = 45.0
    neutral_p2: float = 90.0
    neutral_p3: float = 90.0
    neutral_roll: float = 90.0
    # peace / V sign on the RIGHT hand (only) sends the arm home
    home_gesture_dwell_s: float = 0.6
    home_gesture_refractory_s: float = 3.0


@dataclass
class ControlConfig:
    # "two_hand" (split-screen roles) or "direct" (legacy one-hand mapping)
    scheme: str = "two_hand"
    overlay_scale: float = 1.0
    lost_freeze_s: float = 10.0          # hold on tracking loss, then go home
    # Pose-to-pose ramp speed (home <-> active). Tunable live with Up/Down keys.
    pose_transition_rate_deg_s: float = 5.0
    # Live-control speed multiplier (pitch slew + yaw/roll rates). '='/'-' keys.
    live_speed_scale: float = 1.0
    # Pose-to-pose trajectory shape: "min_jerk" (smooth start/stop, all joints
    # arrive together) or "linear" (legacy constant-rate ramp).
    transition_profile: str = "min_jerk"


@dataclass
class JointMotionConfig:
    """Per-joint motion limits for the host-side pitch profile."""
    rate_deg_s: float = 45.0
    accel_deg_s2: float = 90.0
    jerk_deg_s3: float = 500.0


def _p1_motion() -> "JointMotionConfig":
    # accel/jerk chosen so the S-curve shaping lag (accel/jerk) sits at its
    # 0.12 s max — deliberately slow-and-buttery for chess-precision teleop
    return JointMotionConfig(rate_deg_s=30.0, accel_deg_s2=50.0, jerk_deg_s3=300.0)


def _p2_motion() -> "JointMotionConfig":
    # p2 couples with both p1 and p3 -> lowest acceleration of the three.
    return JointMotionConfig(rate_deg_s=25.0, accel_deg_s2=40.0, jerk_deg_s3=240.0)


def _p3_motion() -> "JointMotionConfig":
    return JointMotionConfig(rate_deg_s=35.0, accel_deg_s2=60.0, jerk_deg_s3=400.0)


@dataclass
class MotionConfig:
    """Selectable host-side pitch smoothing strategy (test one at a time):
      trapezoid  — legacy AccelSlewLimiter baseline
      s_curve    — jerk-limited + target-velocity feedforward (recommended)
      exp_smooth — first-order tracker, never overshoots
      none       — host sends raw (One-Euro-filtered) targets; firmware smooths
    """
    pitch_profile: str = "s_curve"
    exp_track_gain: float = 5.0
    # Dynamic slew (slow near active pose) modulates the velocity cap mid-move,
    # which reads as accel-slow-accel on long sweeps. Off for the new profiles;
    # only the legacy trapezoid uses it when enabled.
    use_dynamic_slew: bool = False
    p1: JointMotionConfig = field(default_factory=_p1_motion)
    p2: JointMotionConfig = field(default_factory=_p2_motion)
    p3: JointMotionConfig = field(default_factory=_p3_motion)


@dataclass
class JointLimitsConfig:
    """Per-joint mechanical software limits (deg). The firmware also clamps to
    these; the host clamps so overlay/targets stay honest."""
    base_min: float = 0.0
    base_max: float = 180.0
    p1_min: float = 0.0
    p1_max: float = 90.0
    # p2 uses a configurable direction; in the current calibration the host
    # sends the direct joint-space angle, which keeps the reversed motor wiring
    # behaving like the earlier arm.
    p2_min: float = 0.0
    p2_max: float = 135.0
    # p3 (upper pitch) full 0-180 range. Neutral hand maps to the active pose
    # p3 (90 = range centre), so takeover has no jump.
    p3_min: float = 0.0
    p3_max: float = 180.0
    roll_min: float = 0.0
    roll_max: float = 180.0


@dataclass
class PoseConfig:
    """A full joint pose (servo deg + gripper open fraction). p2 follows the
    configured host direction, so the same pose data works with either wiring."""
    base_deg: float = 90.0
    p1_deg: float = 45.0
    p2_deg: float = 112.5   # neutral-hand p2 target for the current calibration
    p3_deg: float = 90.0    # centre of the 0-180 p3 range
    roll_deg: float = 90.0
    gripper_open: float = 1.0


def _default_home_pose() -> "PoseConfig":
    # Most energy-efficient parked pose for this arm (user-measured).
    return PoseConfig(base_deg=90.0, p1_deg=98.1, p2_deg=88.2,
                      p3_deg=90.0, roll_deg=90.0, gripper_open=1.0)


@dataclass
class GripperConfig:
    """Gripper servo calibration: the host converts the [0,1] open fraction to
    an absolute servo angle before transmission (open=166.5, closed=36)."""
    open_deg: float = 166.5
    closed_deg: float = 36.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    tcp: TcpConfig = field(default_factory=TcpConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    two_hand: TwoHandConfig = field(default_factory=TwoHandConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    limits: JointLimitsConfig = field(default_factory=JointLimitsConfig)
    active_pose: PoseConfig = field(default_factory=PoseConfig)
    home_pose: PoseConfig = field(default_factory=_default_home_pose)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    config_path: str = ""


def _convert_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None and is_dataclass(field_type):
        return _from_dict(field_type, value or {})
    if origin in (list, tuple):
        args = get_args(field_type)
        item_type = args[0] if args else Any
        return [_convert_value(item_type, item) for item in value]
    return value


def _from_dict(cls: Type[T], values: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for current_field in fields(cls):
        if current_field.name not in values:
            continue
        field_type = type_hints.get(current_field.name, current_field.type)
        kwargs[current_field.name] = _convert_value(field_type, values[current_field.name])
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = _from_dict(AppConfig, raw)
    config.config_path = str(config_path)
    return config


def save_config(config: AppConfig, path: str | Path | None = None) -> None:
    target = Path(path or config.config_path)
    payload = asdict(config)
    payload.pop("config_path", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
