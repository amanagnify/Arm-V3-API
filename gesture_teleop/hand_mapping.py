from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import AppConfig
from filters import apply_signed_deadband, clamp

if TYPE_CHECKING:  # keep this module importable without cv2/mediapipe (unit tests)
    from vision import HandObservation


@dataclass
class TeleopSignals:
    x_offset_norm: float        # [-1,1] horizontal offset from neutral (yaw input)
    height_norm: float          # [0,1] hand height on screen (p3 input)
    depth_norm: float           # [0,1] Project-A depth proxy (p1 input)
    roll_tilt_delta: float      # rad, hand tilt from vertical (yaw-decoupled)
    roll_input_norm: float      # [-1,1] normalized roll input
    gripper_open: float         # [0,1] proportional openness
    finger_curl_norm: float     # [0,1]
    inside_region: bool
    region_xy: tuple[float, float]


# ============================================================================
# Region helpers
# ============================================================================
def _inside_region(x: float, y: float, cfg: AppConfig) -> bool:
    r = cfg.vision.active_region
    return r.x_min <= x <= r.x_max and r.y_min <= y <= r.y_max


def _normalize_region(x: float, y: float, cfg: AppConfig) -> tuple[float, float]:
    r = cfg.vision.active_region
    return clamp((x - r.x_min) / r.width, 0.0, 1.0), clamp((y - r.y_min) / r.height, 0.0, 1.0)


def _curl_to_gripper(finger_curl_norm: float, cfg: AppConfig) -> float:
    """Continuous, proportional open fraction from finger curl. Only the last few
    percent snaps to fully open/closed so the extremes are reliably reachable."""
    lo = clamp(cfg.mapping.gripper_full_open_threshold, 0.0, 0.95)
    hi = clamp(cfg.mapping.gripper_full_close_threshold, lo + 0.01, 1.0)
    span = max(hi - lo, 1e-6)
    if finger_curl_norm <= lo:
        g = 1.0
    elif finger_curl_norm >= hi:
        g = 0.0
    else:
        g = clamp(1.0 - (finger_curl_norm - lo) / span, 0.0, 1.0)
        g = g ** max(cfg.mapping.gripper_exponent, 1e-6)
    if cfg.mapping.gripper_invert:
        g = 1.0 - g
    return g


# ============================================================================
# Signal extraction (hand landmarks -> normalized control signals)
# ============================================================================
def extract_signals(observation: "HandObservation", cfg: AppConfig) -> TeleopSignals:
    center_x, center_y = observation.center_xy
    region_x, region_y = _normalize_region(center_x, center_y, cfg)

    x_offset = clamp(
        (center_x - cfg.vision.neutral_x) / max(cfg.vision.active_region.width * 0.5, 1e-6),
        -1.0, 1.0,
    )
    if cfg.mapping.yaw_invert:
        x_offset = -x_offset

    height_norm = 1.0 - region_y

    # Roll from in-plane hand tilt; subtract a small term proportional to x so a
    # sideways hand shift does not leak into roll (keeps yaw and roll independent).
    roll_tilt_delta = observation.wrist_tilt_metric - cfg.vision.wrist_tilt_reference
    roll_tilt_delta -= cfg.mapping.wrist_yaw_compensation_gain * x_offset
    roll_input_norm = clamp(roll_tilt_delta / max(cfg.mapping.roll_tilt_span_rad, 1e-6), -1.0, 1.0)

    # Depth proxy (Project A): ratio of current metric to the neutral reference.
    depth_ratio = observation.depth_metric / max(cfg.vision.depth_reference, 1e-6)
    # log(ratio) is symmetric (equal reach for equal distance factor either way),
    # so the far side (p1 -> max) isn't compressed; the plain (ratio-1) is linear.
    depth_signal = math.log(max(depth_ratio, 1e-3)) if cfg.mapping.depth_symmetric else (depth_ratio - 1.0)
    depth_dev = cfg.mapping.depth_to_reach_sign * depth_signal * cfg.mapping.depth_gain

    # Precision deadband: the depth proxy (apparent hand size) is noisy, so small
    # deviations from neutral are zeroed -> the arm holds a depth steady despite
    # hand tremor. apply_signed_deadband also rescales, so full reach stays
    # available past the deadband. Applied to the signal (not the roll comp).
    depth_dev = 0.5 * apply_signed_deadband(clamp(depth_dev * 2.0, -1.0, 1.0),
                                            cfg.mapping.depth_deadband)
    depth_norm = 0.5 + depth_dev

    # Roll <-> depth decoupling (feedforward). Rolling the hand changes its
    # apparent size, which leaks into the depth proxy and spikes p1. Add a
    # correction worth `roll_depth_comp_deg` of p1 at full roll, opposing the
    # coupling. depth->p1 is inverted (lower_depth_invert), so d(p1) =
    # -p1_span * d(depth_norm); the sign flips the observed lean back out.
    p1_span = max(cfg.limits.p1_max - cfg.limits.p1_min, 1e-6)
    depth_norm -= (cfg.mapping.roll_depth_comp_deg / p1_span) * roll_input_norm
    depth_norm = clamp(depth_norm, 0.0, 1.0)

    curl_span = max(cfg.vision.grip_closed_reference - cfg.vision.grip_open_reference, 1e-6)
    finger_curl_norm = clamp(
        (observation.finger_curl_metric - cfg.vision.grip_open_reference) / curl_span, 0.0, 1.0
    )
    gripper_open = _curl_to_gripper(finger_curl_norm, cfg)

    return TeleopSignals(
        x_offset_norm=x_offset,
        height_norm=height_norm,
        depth_norm=depth_norm,
        roll_tilt_delta=roll_tilt_delta,
        roll_input_norm=roll_input_norm,
        gripper_open=gripper_open,
        finger_curl_norm=finger_curl_norm,
        inside_region=_inside_region(center_x, center_y, cfg),
        region_xy=(region_x, region_y),
    )


# ============================================================================
# Yaw & roll: integrated-rate controls (deadband + non-linear accel + hold)
# ============================================================================
def integrate_rate(
    prev_angle: float,
    input_norm: float,
    deadband: float,
    exponent: float,
    rate_deg_s: float,
    dt: float,
    lo: float,
    hi: float,
) -> float:
    """Advance an absolute joint angle by a velocity derived from `input_norm`.
    Inside the deadband the velocity is zero, so the joint HOLDS its position
    until an opposite input is given. Acceleration is non-linear: slow near the
    neutral point, faster toward the edges (|input|**exponent)."""
    shaped = apply_signed_deadband(input_norm, deadband)
    if shaped == 0.0:
        return clamp(prev_angle, lo, hi)
    velocity = (abs(shaped) ** max(exponent, 1e-6)) * rate_deg_s
    step = velocity * max(dt, 0.0)
    nxt = prev_angle + (step if shaped > 0.0 else -step)
    return clamp(nxt, lo, hi)


def compute_base_angle(
    prev_base: float, x_offset_norm: float, dt: float, cfg: AppConfig, rate_scale: float = 1.0
) -> float:
    return integrate_rate(
        prev_base, x_offset_norm,
        cfg.mapping.yaw_deadband, cfg.mapping.yaw_exponent,
        cfg.mapping.yaw_rate_deg_s * max(0.0, rate_scale),
        dt, cfg.limits.base_min, cfg.limits.base_max,
    )


def compute_roll_angle(
    prev_roll: float, roll_input_norm: float, dt: float, cfg: AppConfig, rate_scale: float = 1.0
) -> float:
    return integrate_rate(
        prev_roll, roll_input_norm,
        cfg.mapping.roll_deadband, cfg.mapping.roll_exponent,
        cfg.mapping.roll_rate_deg_s * max(0.0, rate_scale),
        dt, cfg.limits.roll_min, cfg.limits.roll_max,
    )


# ============================================================================
# Pitch joints: two switchable methods (direct feature-per-joint, or planar IK)
# ============================================================================
def compute_pitch_targets(
    height_norm: float, depth_norm: float, prev_deg: tuple[float, float, float] | None, cfg: AppConfig
) -> tuple[float, float, float]:
    if cfg.mapping.control_mode == "ik":
        ik = _pitch_from_ik(height_norm, depth_norm, prev_deg, cfg)
        if ik is not None:
            return ik
        # Unreachable target -> fall back to the direct mapping so we never stall.
    return _pitch_direct(height_norm, depth_norm, cfg)


def _apply_p2_flip(p2_joint_deg: float, cfg: AppConfig) -> float:
    """Apply the configured p2 direction. Downstream values stay in servo-space."""
    return 180.0 - p2_joint_deg if cfg.mapping.p2_invert else p2_joint_deg


def _pitch_direct(height_norm: float, depth_norm: float, cfg: AppConfig) -> tuple[float, float, float]:
    """p1 <- depth (reach), p3 <- height, p2 = average of p1 and p3 (then the
    servo flip, so p1=p3=0 commands the p2 servo to 180).

    With `lower_depth_invert` (the default), a SMALLER depth proxy (hand moved
    closer to the camera) drives p1 toward 0, and a LARGER one (hand away)
    drives p1 toward its max."""
    lim = cfg.limits
    d = (1.0 - depth_norm) if cfg.mapping.lower_depth_invert else depth_norm
    p1 = lim.p1_min + d * (lim.p1_max - lim.p1_min)
    p3 = lim.p3_min + height_norm * (lim.p3_max - lim.p3_min)
    p2_joint = clamp(0.5 * (p1 + p3), lim.p2_min, lim.p2_max)
    p2 = _apply_p2_flip(p2_joint, cfg)
    return (clamp(p1, lim.p1_min, lim.p1_max), p2, clamp(p3, lim.p3_min, lim.p3_max))


def reach_height_to_rz(depth_norm: float, height_norm: float, cfg: AppConfig) -> tuple[float, float]:
    w = cfg.workspace
    r = w.target_r_min_mm + depth_norm * (w.target_r_max_mm - w.target_r_min_mm)
    z = w.target_z_min_mm + height_norm * (w.target_z_max_mm - w.target_z_min_mm)
    return r, z


def solve_planar_3r(
    r: float, z: float, prev_q: tuple[float, float, float] | None, cfg: AppConfig
) -> tuple[float, float, float] | None:
    """Redundant 3R planar IK. Returns planar joint angles (deg) q1,q2,q3 relative
    to the shoulder frame, resolving the extra DOF by sampling the end-effector
    ('tool') pitch and minimizing (orientation preference + continuity) cost.
    Returns None if the point is unreachable for every sampled tool angle."""
    k = cfg.kinematics
    l1, l2, l3 = k.l1_mm, k.l2_mm, k.l3_mm
    tx = r - k.shoulder_r_offset_mm
    ty = z - k.shoulder_z_offset_mm

    best_cost = None
    best_q: tuple[float, float, float] | None = None

    phi = k.tool_angle_min_deg
    while phi <= k.tool_angle_max_deg + 1e-9:
        pr = math.radians(phi)
        wx = tx - l3 * math.cos(pr)
        wy = ty - l3 * math.sin(pr)
        d2 = wx * wx + wy * wy
        d = math.sqrt(d2)
        if 1e-6 < d <= (l1 + l2) and d >= abs(l1 - l2):
            cos_q2 = clamp((d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
            q2 = math.acos(cos_q2)  # elbow-down branch
            q1 = math.atan2(wy, wx) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
            q3 = pr - q1 - q2
            q = (math.degrees(q1), math.degrees(q2), math.degrees(q3))
            cost = k.orientation_preference_weight * abs(phi - k.preferred_tool_angle_deg)
            if prev_q is not None:
                cost += k.continuity_weight * sum(abs(a - b) for a, b in zip(q, prev_q))
            if best_cost is None or cost < best_cost:
                best_cost, best_q = cost, q
        phi += k.tool_angle_step_deg

    return best_q


def _pitch_from_ik(
    height_norm: float, depth_norm: float, prev_deg: tuple[float, float, float] | None, cfg: AppConfig
) -> tuple[float, float, float] | None:
    """EXPERIMENTAL. Maps the hand to an (r,z) target, solves the planar IK, then
    places each planar joint angle into its servo range around the neutral pose.
    The precise servo-zero convention is firmware calibration; this affine offset
    (limit mid-point + planar angle) is a tunable starting point for comparison."""
    r, z = reach_height_to_rz(depth_norm, height_norm, cfg)
    q = solve_planar_3r(r, z, None, cfg)
    if q is None:
        return None
    lim = cfg.limits
    p1 = clamp(0.5 * (lim.p1_min + lim.p1_max) + q[0], lim.p1_min, lim.p1_max)
    p2_joint = clamp(0.5 * (lim.p2_min + lim.p2_max) + q[1], lim.p2_min, lim.p2_max)
    p3 = clamp(0.5 * (lim.p3_min + lim.p3_max) + q[2], lim.p3_min, lim.p3_max)
    return (p1, _apply_p2_flip(p2_joint, cfg), p3)
