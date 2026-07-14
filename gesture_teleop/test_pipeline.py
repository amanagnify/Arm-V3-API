"""Headless unit tests for the teleop control pipeline.

Run:  python test_pipeline.py
Exercises the pure logic (no camera / no MediaPipe frames) end to end.
"""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import protocol
import hand_mapping as hm
from config import load_config
from filters import (
    AccelSlewLimiter,
    ExpSmoothLimiter,
    OneEuroFilter,
    SCurveLimiter,
    min_jerk_s,
)

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -> ' + extra if extra and not cond else ''}")
    if not cond:
        FAILS.append(name)


def fake_signals(**kw):
    base = dict(
        x_offset_norm=0.0, height_norm=0.5, depth_norm=0.5, roll_tilt_delta=0.0,
        roll_input_norm=0.0, gripper_open=1.0, finger_curl_norm=0.0,
        inside_region=True, region_xy=(0.5, 0.5),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_protocol():
    print("protocol V3 round-trip (all six channels are absolute angles)")
    pkt = protocol.build_packet(7, "A", 90, 45, 67.5, 180, 90, 101.25)
    check("length 19", len(pkt) == protocol.PACKET_LEN)
    d = protocol.parse_packet(pkt)
    check("seq/mode", d["seq"] == 7 and d["mode"] == "A")
    check("angles round-trip", all(abs(d[k] - v) < 0.02 for k, v in
          {"base": 90, "p1": 45, "p2": 67.5, "p3": 180, "roll": 90,
           "gripper": 101.25}.items()), str(d))
    bad = bytearray(pkt); bad[9] ^= 0xFF
    check("checksum rejects corruption", protocol.parse_packet(bytes(bad)) is None)
    bad_mode = bytearray(pkt); bad_mode[4] = ord("S")
    bad_mode[-1] = 0
    for b in bad_mode[:-1]:
        bad_mode[-1] ^= b
    check("mode S rejected", protocol.parse_packet(bytes(bad_mode)) is None)
    try:
        protocol.build_packet(0, "S", 0, 0, 0, 0, 0, 0)
        valid_mode_guard = False
    except ValueError:
        valid_mode_guard = True
    check("builder refuses invalid modes", valid_mode_guard)
    check("clamps over-range angle", protocol.parse_packet(
        protocol.build_packet(0, "A", 999, -50, 0, 0, 0, 0))["base"] == 180.0)


def test_integrator(cfg):
    print("yaw/roll rate integrator (deadband + accel + hold)")
    b = 90.0
    check("hold inside deadband", hm.compute_base_angle(b, 0.05, 0.033, cfg) == 90.0)
    moved = hm.compute_base_angle(b, 1.0, 0.033, cfg)
    check("moves on full input", moved > 90.0)
    # non-linear accel: small input moves much less than large input
    small = hm.compute_base_angle(90.0, 0.4, 0.033, cfg) - 90.0
    large = hm.compute_base_angle(90.0, 1.0, 0.033, cfg) - 90.0
    check("accel non-linear (small << large)", small * 2 < large, f"{small:.3f} vs {large:.3f}")
    # hold-until-opposite then reverse
    held = hm.compute_base_angle(moved, 0.0, 0.033, cfg)
    check("holds after release", abs(held - moved) < 1e-9)
    back = hm.compute_base_angle(moved, -1.0, 0.033, cfg)
    check("reverses on opposite input", back < moved)
    # live-speed scale slows the same input down
    slow = hm.compute_base_angle(90.0, 1.0, 0.033, cfg, rate_scale=0.5) - 90.0
    check("rate_scale halves speed", abs(slow * 2 - large) < 1e-6, f"{slow:.4f} vs {large:.4f}")
    # clamps to limits
    hi = 90.0
    for _ in range(2000):
        hi = hm.compute_base_angle(hi, 1.0, 0.033, cfg)
    check("clamps to base_max", hi <= cfg.limits.base_max + 1e-6)


def test_one_euro():
    print("One-Euro filter")
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
    import random
    random.seed(0)
    out = 0.0
    for _ in range(200):
        out = f.update(0.5 + random.uniform(-0.02, 0.02), 1 / 60)
    check("smooths noise around mean", abs(out - 0.5) < 0.02, f"{out:.3f}")
    f2 = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    y = 0.0
    for i in range(60):
        y = f2.update(i / 60.0, 1 / 60)
    check("tracks a ramp (lag < 0.25)", abs(y - 59 / 60.0) < 0.25, f"{y:.3f}")


def test_accel_slew(cfg):
    print("AccelSlewLimiter (gentle start from standstill, bounded speed)")
    s = AccelSlewLimiter(cfg.filters.joint_rate_deg_s, cfg.filters.joint_accel_deg_s2, 0.0)
    dt = 1 / 30.0
    first = s.update(90.0, dt)
    plain_step = cfg.filters.joint_rate_deg_s * dt
    check("first step is gentle (accel-limited)", first < plain_step * 0.25,
          f"{first:.3f} vs plain {plain_step:.3f}")
    prev = first
    max_step = 0.0
    peak = first
    for _ in range(600):
        v = s.update(90.0, dt)
        max_step = max(max_step, abs(v - prev))
        peak = max(peak, v)
        prev = v
    check("converges to target", abs(prev - 90.0) < 1.5, f"{prev:.2f}")
    check("respects velocity cap", max_step <= plain_step + 0.05, f"{max_step:.3f}")
    check("no big overshoot", peak < 93.0, f"{peak:.2f}")


def test_gripper(cfg):
    print("proportional gripper")
    vals = [hm._curl_to_gripper(c / 20.0, cfg) for c in range(21)]
    inv = cfg.mapping.gripper_invert
    if inv:
        # inverted: open fist -> open (1.0), open palm -> closed (0.0)
        check("monotonic increasing with curl (inverted)",
              all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)))
        check("closed at low curl (open palm)", vals[0] < 0.05)
        check("open at high curl (fist)", vals[-1] > 0.95)
    else:
        check("monotonic decreasing with curl",
              all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)))
        check("open at low curl", vals[0] > 0.95)
        check("closed at high curl", vals[-1] < 0.05)
    mids = vals[5:16]
    check("continuous mid-range (not 2-state)", len(set(round(v, 3) for v in mids)) > 5)

    import main as m
    pose = m.Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)
    check("open fraction -> open_deg (166.5) servo",
          abs(pose.command("A", cfg.gripper).gripper_deg - cfg.gripper.open_deg) < 1e-6)
    pose.grip = 0.0
    check("closed fraction -> closed_deg (36) servo",
          abs(pose.command("A", cfg.gripper).gripper_deg - cfg.gripper.closed_deg) < 1e-6)
    pose.grip = 0.5
    mid = 0.5 * (cfg.gripper.open_deg + cfg.gripper.closed_deg)
    check("half open -> mid servo angle",
          abs(pose.command("A", cfg.gripper).gripper_deg - mid) < 1e-6)


def test_direct_pitch(cfg):
    print("direct pitch mapping ranges (depth inverted -> p1, p2 servo flipped)")
    cfg.mapping.control_mode = "direct_joint"
    cfg.mapping.lower_depth_invert = True
    # hand far (depth_norm=0) reaches out -> p1 max; low (height=0) -> p3 min
    p1, p2, p3 = hm.compute_pitch_targets(0.0, 0.0, None, cfg)
    check("far+low: p1=90, p3=p3_min",
          abs(p1 - 90) < 1e-6 and abs(p3 - cfg.limits.p3_min) < 1e-6, f"p1={p1} p3={p3}")
    check("p2 = 180 - avg(p1,p3) servo-flipped",
          abs(p2 - (180.0 - 0.5 * (p1 + p3))) < 1e-6, str(p2))
    # hand near (depth_norm=1) -> p1 folds toward 0
    p1n, _, _ = hm.compute_pitch_targets(0.0, 1.0, None, cfg)
    check("near: p1 folds to 0", abs(p1n) < 1e-6, str(p1n))
    # hand high (height=1) -> p3 max
    _, _, p3h = hm.compute_pitch_targets(1.0, 0.5, None, cfg)
    check("high: p3=p3_max", abs(p3h - cfg.limits.p3_max) < 1e-6, str(p3h))
    # neutral (0.5, 0.5) must map to the active pose so takeover has no jump.
    p1m, p2m, p3m = hm.compute_pitch_targets(0.5, 0.5, None, cfg)
    mid_p3 = 0.5 * (cfg.limits.p3_min + cfg.limits.p3_max)
    check("neutral -> active (p1=45, p3=range centre, aligned)",
          abs(p1m - 45) < 1e-6 and abs(p3m - mid_p3) < 1e-6, f"p1={p1m} p3={p3m}")
    check("neutral p3 == active pose p3", abs(p3m - cfg.active_pose.p3_deg) < 1e-6, str(p3m))
    check("neutral maps to active pose p2",
          abs(p2m - cfg.active_pose.p2_deg) < 1e-6, f"{p2m} vs {cfg.active_pose.p2_deg}")
    cfg.mapping.p2_invert = False
    p1u, p2u, p3u = hm.compute_pitch_targets(0.5, 0.5, None, cfg)
    check("p2_invert=False keeps joint space (= avg(p1,p3))",
          abs(p2u - 0.5 * (p1u + p3u)) < 1e-6, str(p2u))
    cfg.mapping.p2_invert = True


def test_depth_response(cfg):
    print("depth response: symmetric reach + linear-uniform option")
    from types import SimpleNamespace

    def p1_of(ratio):  # p1 target for a hand size = ratio * neutral reference
        obs = SimpleNamespace(
            center_xy=(cfg.vision.neutral_x, cfg.vision.neutral_y),
            wrist_tilt_metric=cfg.vision.wrist_tilt_reference,  # no roll
            depth_metric=cfg.vision.depth_reference * ratio,
            finger_curl_metric=cfg.vision.grip_open_reference,
        )
        return hm.compute_pitch_targets(0.5, hm.extract_signals(obs, cfg).depth_norm, None, cfg)[0]

    saved = (cfg.mapping.depth_deadband, cfg.mapping.depth_gain, cfg.mapping.depth_symmetric)
    cfg.mapping.depth_deadband = 0.0

    # neutral hand -> p1 at range centre (45)
    check("neutral depth -> p1 45", abs(p1_of(1.0) - 45.0) < 1e-6, str(p1_of(1.0)))

    # SYMMETRIC (log): the FULL 0..90 range is reachable with modest, symmetric
    # hand travel (halving/doubling apparent size), unlike the lopsided linear map.
    cfg.mapping.depth_symmetric = True
    cfg.mapping.depth_gain = 0.85
    far = p1_of(0.5)    # hand half apparent size (receded)
    near = p1_of(2.0)   # hand double apparent size (approached)
    check("symmetric: far hand reaches p1 max (90)", far > 89.0, f"{far:.1f}")
    check("symmetric: near hand reaches p1 min (0)", near < 1.0, f"{near:.1f}")
    check("symmetric: equal DISTANCE factor is symmetric about 45",
          abs((far - 45.0) + (near - 45.0)) < 1.0, f"{far:.1f}/{near:.1f}")

    # LINEAR is lopsided: the far side compresses (this is the bug the log fixes)
    cfg.mapping.depth_symmetric = False
    check("linear: far side compressed (bug the symmetric map fixes)",
          p1_of(0.5) < far - 5.0, f"{p1_of(0.5):.1f}")

    cfg.mapping.depth_deadband, cfg.mapping.depth_gain, cfg.mapping.depth_symmetric = saved


def test_output_clamp(cfg):
    print("pitch output clamp (no negative / over-max from s-curve overshoot)")
    import main as m
    from filters import SCurveLimiter

    bounds = m._pitch_bounds(cfg)
    check("p1 bounds = joint limits", bounds["p1"] == (cfg.limits.p1_min, cfg.limits.p1_max))
    check("p2 bounds are servo-flipped",
          bounds["p2"] == (180.0 - cfg.limits.p2_max, 180.0 - cfg.limits.p2_min))

    # simulate an overshoot below 0 and confirm it is clamped + state reset
    lim = SCurveLimiter(40.0, 70.0, 400.0, 5.0)
    lim.value, lim.velocity = -12.0, -30.0
    out = m._clamp_limiter(lim, lim.value, 0.0, 90.0)
    check("negative overshoot clamped to 0", out == 0.0, str(out))
    check("velocity zeroed at the limit (anti-windup)", lim.velocity == 0.0 and lim.value == 0.0)

    # a hard step target driven to the p1 max never leaves [0,90] frame to frame
    lim2 = SCurveLimiter(cfg.motion.p1.rate_deg_s, cfg.motion.p1.accel_deg_s2,
                         cfg.motion.p1.jerk_deg_s3, 45.0)
    worst = 45.0
    for _ in range(400):
        v = m._clamp_limiter(lim2, lim2.update(90.0, 1 / 30.0), 0.0, 90.0)
        worst = max(worst, abs(v - 45.0))
        check_in = (v >= -1e-9 and v <= 90.0 + 1e-9)
        if not check_in:
            break
    check("stays within [0,90] tracking a step to the limit", check_in and worst > 40.0)


def test_roll_depth_decouple(cfg):
    print("roll->depth decoupling + low-curl depth hold")
    from types import SimpleNamespace

    def obs(tilt):
        return SimpleNamespace(
            center_xy=(cfg.vision.neutral_x, cfg.vision.neutral_y),
            wrist_tilt_metric=cfg.vision.wrist_tilt_reference + tilt,
            depth_metric=cfg.vision.depth_reference,  # exactly neutral depth
            finger_curl_metric=cfg.vision.grip_open_reference,
        )

    base = hm.extract_signals(obs(0.0), cfg).depth_norm
    check("no roll comp at zero roll (depth stays neutral)", abs(base - 0.5) < 1e-6, f"{base:.3f}")
    full = hm.extract_signals(obs(cfg.mapping.roll_tilt_span_rad), cfg).depth_norm
    expected = cfg.mapping.roll_depth_comp_deg / (cfg.limits.p1_max - cfg.limits.p1_min)
    check("full roll compensates ~roll_depth_comp_deg of p1",
          abs(abs(full - 0.5) - expected) < 1e-3, f"shift={abs(full - 0.5):.4f} vs {expected:.4f}")
    check("comp is odd in roll (left/right opposite)",
          abs((hm.extract_signals(obs(-0.3), cfg).depth_norm - 0.5)
              + (hm.extract_signals(obs(0.3), cfg).depth_norm - 0.5)) < 1e-6)
    # grasp depth hold now triggers early (10% curl) with clean hysteresis
    check("grip hold trigger <= 0.10 curl", cfg.filters.depth_hold_curl_threshold <= 0.10 + 1e-9)
    check("release below trigger (valid hysteresis)",
          cfg.filters.depth_hold_release_curl_threshold < cfg.filters.depth_hold_curl_threshold)


def _fake_hand(cfg, x=0.5, y=0.5, size=None, curl=0.0, tilt=0.0, peace=False):
    """Minimal HandObservation stand-in for the two-hand controller."""
    import numpy as np
    from types import SimpleNamespace
    lm = np.zeros((21, 3), dtype=float)
    if peace:  # index+middle extended (tip above pip), ring+pinky curled
        for tip, pip in ((8, 6), (12, 10)):
            lm[pip, 1], lm[tip, 1] = 0.5, 0.4
        for tip, pip in ((16, 14), (20, 18)):
            lm[pip, 1], lm[tip, 1] = 0.5, 0.6
    curl_metric = cfg.vision.grip_open_reference + curl * (
        cfg.vision.grip_closed_reference - cfg.vision.grip_open_reference)
    return SimpleNamespace(
        center_xy=(x, y), depth_metric=size if size is not None else 0.10,
        finger_curl_metric=curl_metric, wrist_tilt_metric=tilt,
        normalized_landmarks=lm, pixel_landmarks=np.zeros((21, 2), np.int32),
        confidence=1.0,
    )


def test_two_hand(cfg):
    print("two-hand: roles, neutral-pose N, locked no-drift + out-of-view lock, R/L height, grip, peace-home")
    import main as m
    from two_hand_control import TwoHandController, is_peace_sign

    th = cfg.two_hand
    pose = m.Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)
    dt = 1 / 30.0
    t = 0.0

    # --- role assignment
    l, r = TwoHandController(cfg).assign_roles([_fake_hand(cfg, x=0.3)], cfg)
    check("single hand left half -> LEFT role", l is not None and r is None)
    tc0 = TwoHandController(cfg)
    l, r = tc0.assign_roles([_fake_hand(cfg, x=0.7)], cfg)
    check("single hand right half -> RIGHT role", r is not None and l is None)
    l, r = tc0.assign_roles([_fake_hand(cfg, x=0.48)], cfg)  # hysteresis keeps RIGHT
    check("midline crossing kept by hysteresis", r is not None and l is None)
    l, r = tc0.assign_roles([_fake_hand(cfg, x=0.6), _fake_hand(cfg, x=0.2)], cfg)
    check("two hands -> leftmost L, rightmost R",
          l is not None and r is not None and l.center_xy[0] < r.center_xy[0])

    # --- CORR4: neutral set ONLY by key (set_anchors) drives the arm to the
    #     NEUTRAL POSE. Right N -> p1=neutral_p1, p2=neutral_p2 at the anchor.
    tc2 = TwoHandController(cfg)
    rhand = _fake_hand(cfg, x=0.75, y=0.5, size=0.12)
    tc2.set_anchors(None, rhand, pose, cfg)          # <- the R key
    f = tc2.update([rhand], pose, t, cfg)
    check("CORR4 right N -> anchor gives p1=neutral_p1, p2=neutral_p2, yaw 0",
          abs(f.p1 - th.neutral_p1) < 1e-9 and abs(f.p2 - th.neutral_p2) < 1e-9
          and abs(f.yaw_input) < 1e-9,
          f"p1={f.p1} p2={f.p2} yaw={f.yaw_input}")
    # Left N -> p3=neutral_p3 and roll SNAPPED to neutral_roll (roll is rate-based)
    pose_l = m.Pose(90.0, 45.0, 90.0, 120.0, 20.0, 1.0)
    tc_l = TwoHandController(cfg)
    lhand = _fake_hand(cfg, x=0.25, y=0.5)
    tc_l.set_anchors(lhand, None, pose_l, cfg)       # <- the L key
    fL = tc_l.update([lhand], pose_l, t, cfg)
    check("CORR4 left N -> p3=neutral_p3 and roll=neutral_roll",
          abs(fL.p3 - th.neutral_p3) < 1e-9 and abs(pose_l.roll - th.neutral_roll) < 1e-9,
          f"p3={fL.p3} roll={pose_l.roll}")
    # engagement/anchor does NOT auto-move: a hand that never got a key keeps default anchor
    tc_na = TwoHandController(cfg)
    ax0 = tc_na.right.anchor_x
    tc_na.update([_fake_hand(cfg, x=0.95, y=0.2, size=0.2)], pose, t, cfg)
    check("no auto re-anchor (anchor only moves on key)", tc_na.right.anchor_x == ax0)

    # CORR3: frame exposes separate R/L heights for the UI bars
    fh = tc2.update([_fake_hand(cfg, x=0.75, y=0.30, size=0.12),
                     _fake_hand(cfg, x=0.25, y=0.30)], pose, t, cfg)
    check("CORR3 frame exposes R height and L height",
          fh.r_height > 0.5 and fh.l_height > 0.5, f"r={fh.r_height:.2f} l={fh.l_height:.2f}")

    # move hand closer (bigger) -> p1 decreases (lower_depth_invert)
    f = tc2.update([_fake_hand(cfg, x=0.75, y=0.5, size=0.16)], pose, t, cfg)
    check("hand closer -> p1 decreases", f.p1 < th.neutral_p1, f"{f.p1:.1f}")

    # --- fist -> PAUSE p1/p2 (+right yaw); open/close/open to RESUME
    for _ in range(int(th.lock_dwell_s / dt) + 3):
        f = tc2.update([_fake_hand(cfg, x=0.85, y=0.4, size=0.16, curl=0.9)], pose, t, cfg)
        t += dt
    check("fist pauses right (locked)", f.right_locked and not f.right_engaged)
    check("paused: p1/p2 hold (None) and right yaw contributes 0",
          f.p1 is None and f.p2 is None and abs(f.yaw_input) < 1e-9)
    # a plain OPEN must NOT resume (needs open->close->open)
    for _ in range(4):
        f = tc2.update([_fake_hand(cfg, x=0.85, y=0.4, size=0.16, curl=0.0)], pose, t, cfg)
        t += dt
    check("plain open does NOT resume", f.right_locked)
    f = tc2.update([_fake_hand(cfg, x=0.85, curl=0.9)], pose, t, cfg)      # close
    f = tc2.update([_fake_hand(cfg, x=0.85, curl=0.0)], pose, t, cfg)      # open -> resume
    check("open/close/open resumes", not tc2.right.locked and f.right_engaged)

    # --- CORR2: while paused, p1/p2 are HELD through the real drive stack and
    #     do NOT drift to a limit (regression: target==output integrator runaway).
    #     CORR1: lock survives the hand leaving view; a plain OPEN on return does
    #     NOT resume (a fresh OPEN->CLOSE->OPEN is required).
    filt = m._build_filters(cfg)
    dpose = m.Pose(90.0, 30.0, 70.0, 90.0, 90.0, 1.0)
    tcd = TwoHandController(cfg)
    tcd.set_anchors(None, _fake_hand(cfg, x=0.75, y=0.5, size=0.12), dpose, cfg)
    for _ in range(5):  # engage & move
        fr = tcd.update([_fake_hand(cfg, x=0.80, y=0.35, size=0.16)], dpose, t, cfg)
        m._drive_two_hand(fr, dt, cfg, filt, dpose, 1.0)
        t += dt
    for _ in range(int(th.lock_dwell_s / dt) + 3):  # fist -> lock
        fr = tcd.update([_fake_hand(cfg, x=0.80, y=0.35, size=0.16, curl=0.9)], dpose, t, cfg)
        m._drive_two_hand(fr, dt, cfg, filt, dpose, 1.0)
        t += dt
    p1_lock, p2_lock = dpose.p1, dpose.p2
    for i in range(200):  # fist present for 100 frames, then OUT OF VIEW for 100
        hi = [] if i >= 100 else [_fake_hand(cfg, x=0.80, y=0.35, size=0.16, curl=0.9)]
        m._drive_two_hand(tcd.update(hi, dpose, t, cfg), dt, cfg, filt, dpose, 1.0)
        t += dt
    check("CORR2 locked p1 does not drift", abs(dpose.p1 - p1_lock) < 0.5,
          f"{p1_lock:.2f} -> {dpose.p1:.2f}")
    check("CORR2 locked p2 does not drift", abs(dpose.p2 - p2_lock) < 0.5,
          f"{p2_lock:.2f} -> {dpose.p2:.2f}")
    check("CORR1 still locked after 100 frames out of view", tcd.right.locked)
    for _ in range(6):  # returns OPEN
        tcd.update([_fake_hand(cfg, x=0.80, y=0.5, size=0.12, curl=0.0)], dpose, t, cfg)
        t += dt
    check("CORR1 plain open on return does NOT unlock", tcd.right.locked)
    tcd.update([_fake_hand(cfg, x=0.80, curl=0.9)], dpose, t, cfg)          # close
    tcd.update([_fake_hand(cfg, x=0.80, curl=0.0)], dpose, t, cfg)          # open
    check("CORR1 open->close->open after return resumes", not tcd.right.locked)

    # --- dual-hand yaw sums; small deflections inside deadband give zero
    tc3 = TwoHandController(cfg)
    lh = _fake_hand(cfg, x=0.25, y=0.5)
    rh = _fake_hand(cfg, x=0.75, y=0.5, size=0.1)
    tc3.set_anchors(lh, rh, pose, cfg)          # key both
    span = th.yaw_deflection_span
    both_right = [_fake_hand(cfg, x=0.25 + 0.8 * span, y=0.5),
                  _fake_hand(cfg, x=0.75 + 0.8 * span, y=0.5, size=0.1)]
    f = tc3.update(both_right, pose, t, cfg)
    check("both hands deflected -> yaw contributions sum", f.yaw_input > 0.5, f"{f.yaw_input:.2f}")
    tiny = [_fake_hand(cfg, x=0.25 + 0.05 * span, y=0.5),
            _fake_hand(cfg, x=0.75 + 0.05 * span, y=0.5, size=0.1)]
    for _ in range(45):  # let the input filters settle onto the tiny deflection
        f = tc3.update(tiny, pose, t, cfg)
        t += dt
    check("tiny deflections inside deadband -> zero yaw", abs(f.yaw_input) < 1e-9)

    # --- left hand: p3 tracks height and is NOT frozen by grip; grip natural
    f = tc3.update([_fake_hand(cfg, x=0.25, y=0.30), both_right[1]], pose, t, cfg)
    check("left hand up -> p3 rises", f.p3 is not None and f.p3 > pose.p3, str(f.p3))
    f = tc3.update([_fake_hand(cfg, x=0.25, y=0.30, curl=0.8), both_right[1]], pose, t, cfg)
    check("gripping does NOT freeze p3 (still tracks)", f.p3 is not None, str(f.p3))
    f = tc3.update([_fake_hand(cfg, x=0.25, y=0.5, curl=0.0), both_right[1]], pose, t, cfg)
    check("left OPEN hand -> gripper OPEN (natural)", f.grip is not None and f.grip > 0.99, str(f.grip))
    f = tc3.update([_fake_hand(cfg, x=0.25, y=0.5, curl=1.0), both_right[1]], pose, t, cfg)
    check("left CLOSED fist -> gripper CLOSED (natural)", f.grip is not None and f.grip < 0.01, str(f.grip))

    # --- RIGHT peace / V sign -> home (left peace does nothing); refractory
    check("peace heuristic true", is_peace_sign(_fake_hand(cfg, peace=True)))
    check("open palm not peace", not is_peace_sign(_fake_hand(cfg)))
    tc4 = TwoHandController(cfg)
    got = False
    for _ in range(int(th.home_gesture_dwell_s / dt) + 4):
        f = tc4.update([_fake_hand(cfg, x=0.75, peace=True)], pose, t, cfg)
        got = got or f.home_requested
        t += dt
    check("dwelled RIGHT peace requests home", got)
    f = tc4.update([_fake_hand(cfg, x=0.75, peace=True)], pose, t, cfg)
    check("refractory blocks immediate retrigger", not f.home_requested)
    tc5 = TwoHandController(cfg)
    got_l = False
    for _ in range(int(th.home_gesture_dwell_s / dt) + 4):
        f = tc5.update([_fake_hand(cfg, x=0.25, peace=True)], pose, t, cfg)  # LEFT half
        got_l = got_l or f.home_requested
        t += dt
    check("LEFT peace does NOT go home (right only)", not got_l)

    # --- SMOOTHNESS: a short tracking flicker must NOT stall the motion.
    # Regression: the absent-frame limiter reset zeroed velocity mid-move,
    # then the accumulated error caused a sudden catch-up jump.
    filt2 = m._build_filters(cfg)
    spose = m.Pose(90.0, 45.0, 90.0, 90.0, 90.0, 1.0)
    tcf = TwoHandController(cfg)
    tcf.set_anchors(None, _fake_hand(cfg, x=0.75, y=0.60, size=0.12), spose, cfg)
    deltas = []
    yy = 0.60
    for i in range(90):
        yy -= 0.004                              # steady upward hand motion
        hands_i = [] if i in (60, 61) else [_fake_hand(cfg, x=0.75, y=yy, size=0.12)]
        fr = tcf.update(hands_i, spose, t, cfg)
        before = spose.p2
        m._drive_two_hand(fr, dt, cfg, filt2, spose, 1.0)
        deltas.append(abs(spose.p2 - before))
        t += dt
    cruise = sum(deltas[50:60]) / 10.0
    check("dropout frames coast (velocity NOT zeroed)",
          min(deltas[60], deltas[61]) > 0.4 * cruise,
          f"cruise={cruise:.3f} flicker={deltas[60]:.3f},{deltas[61]:.3f}")
    check("no catch-up spike after the flicker",
          max(deltas[62:75]) < 1.6 * max(cruise, 1e-6),
          f"post={max(deltas[62:75]):.3f} cruise={cruise:.3f}")

    # --- SMOOTHNESS: raw hand jitter must not reach the joint targets
    # (One-Euro input filtering; unfiltered p1 would swing ~4.6 deg p2p here)
    tcn = TwoHandController(cfg)
    npose = m.Pose(90.0, 45.0, 90.0, 90.0, 90.0, 1.0)
    tcn.set_anchors(None, _fake_hand(cfg, x=0.75, y=0.5, size=0.12), npose, cfg)
    p1_targets, yaw_ok = [], True
    for i in range(90):
        s = 1.0 + (0.03 if i % 2 == 0 else -0.03)     # +/-3% size noise
        xj = 0.75 + (0.004 if i % 2 == 0 else -0.004)  # +/- pixel-level x noise
        fr = tcn.update([_fake_hand(cfg, x=xj, y=0.5, size=0.12 * s)], npose, t, cfg)
        p1_targets.append(fr.p1)
        yaw_ok = yaw_ok and abs(fr.yaw_input) < 1e-9
        t += dt
    tail = p1_targets[50:]
    p2p = max(tail) - min(tail)
    check("input noise filtered: p1 target wobble < 1 deg p2p", p2p < 1.0, f"{p2p:.2f} deg")
    check("x noise never leaks into yaw (deadband + filter)", yaw_ok)

    # --- FIRMWARE COMPATIBILITY: two-hand drives the SAME V3 wire packet, and
    # every channel a two-hand session can produce stays inside firmware clamps.
    filters = m._build_filters(cfg)
    p = m.Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)
    tc6 = TwoHandController(cfg)
    lh = _fake_hand(cfg, x=0.20, y=0.15, size=0.10, curl=1.0)      # extremes
    rh = _fake_hand(cfg, x=0.98, y=0.05, size=0.30)
    tc6.set_anchors(lh, rh, p, cfg)
    worst_ok = True
    tt = 0.0
    for _ in range(120):
        fr = tc6.update([lh, rh], p, tt, cfg)
        m._drive_two_hand(fr, dt, cfg, filters, p, 1.0)
        cmd = p.command("A", cfg.gripper)
        _, tgt = fw_apply(cmd.to_packet(), [0.0] * 6)   # firmware intake (parity port)
        # firmware accepted the packet AND its clamps didn't have to change our angles
        want = [p.base, p.p1, p.p2, p.p3, p.roll, cmd.gripper_deg]
        if any(abs(tgt[i] - want[i]) > 0.05 for i in range(6)):
            worst_ok = False
            break
        tt += dt
    check("two-hand poses stay within firmware clamps (V3 wire unchanged)", worst_ok)


def _fk(q_deg, cfg):
    k = cfg.kinematics
    q1, q2, q3 = (math.radians(a) for a in q_deg)
    x = k.shoulder_r_offset_mm + k.l1_mm * math.cos(q1) + k.l2_mm * math.cos(q1 + q2) + k.l3_mm * math.cos(q1 + q2 + q3)
    y = k.shoulder_z_offset_mm + k.l1_mm * math.sin(q1) + k.l2_mm * math.sin(q1 + q2) + k.l3_mm * math.sin(q1 + q2 + q3)
    return x, y


def test_ik(cfg):
    print("planar 3R IK <-> FK round-trip")
    r, z = hm.reach_height_to_rz(0.5, 0.5, cfg)
    q = hm.solve_planar_3r(r, z, None, cfg)
    check("solves reachable target", q is not None, f"r={r:.1f} z={z:.1f}")
    if q is not None:
        x, y = _fk(q, cfg)
        check("FK returns to target", math.hypot(x - r, y - z) < 1.0, f"err={math.hypot(x-r,y-z):.3f}mm")
    far = hm.solve_planar_3r(100000.0, 0.0, None, cfg)
    check("unreachable -> None", far is None)


def test_ramp(cfg):
    print("pose-to-pose ramp engine (slow, bounded rate)")
    import main as m
    pose = m.Pose(90.0, 98.1, 88.2, 90.0, 90.0, 1.0)   # home
    target = m.Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)  # active
    rate, dt = 5.0, 0.1
    grip_rate = rate / (cfg.gripper.closed_deg - cfg.gripper.open_deg)
    prev_p1 = pose.p1
    arrived = m._ramp_pose_toward(pose, target, rate, grip_rate, dt)
    check("first step bounded by rate*dt", abs(pose.p1 - prev_p1) <= rate * dt + 1e-9,
          f"{abs(pose.p1 - prev_p1):.3f}")
    check("not arrived immediately", not arrived)
    steps = 1
    while steps < 3000 and not m._ramp_pose_toward(pose, target, rate, grip_rate, dt):
        steps += 1
    expected = abs(98.1 - 45.0) / (rate * dt)  # ~106 steps for the longest joint
    check("arrives in ~expected time", abs(steps - expected) < 15, f"{steps} vs {expected:.0f}")
    check("lands on target", abs(pose.p1 - 45.0) < 0.6 and abs(pose.p2 - 112.5) < 0.6,
          f"p1={pose.p1:.2f} p2={pose.p2:.2f}")


def test_depth_hold(cfg):
    print("depth-hold: dwell gating, pre-gesture snapshot, timed resume")
    import main
    dt = 1 / 30.0
    filters = main._build_filters(cfg)
    active = main._active_pose(cfg)
    pose = main.Pose(active.base, active.p1, active.p2, active.p3, active.roll, active.grip)
    cs = main.ControlState(prev_pitch=(active.p1, active.p2, active.p3))
    t = 0.0

    # open hand, depth ~0.5 -> establishes filter, no hold
    for _ in range(10):
        main._drive_from_hand(fake_signals(depth_norm=0.5, finger_curl_norm=0.0), dt, t, cfg, filters, pose, cs)
        t += dt
    check("no hold while open", not cs.depth_hold)
    depth_at_close = filters["depth"].value

    # brief curl flicker (shorter than the dwell) must NOT trigger a hold
    for _ in range(4):  # ~0.13 s < hold_engage_dwell_s
        main._drive_from_hand(fake_signals(depth_norm=0.6, finger_curl_norm=0.9), dt, t, cfg, filters, pose, cs)
        t += dt
    check("brief flicker does not engage hold", not cs.depth_hold)
    for _ in range(6):  # reopen to re-arm cleanly
        main._drive_from_hand(fake_signals(depth_norm=0.5, finger_curl_norm=0.0), dt, t, cfg, filters, pose, cs)
        t += dt
    depth_at_close = filters["depth"].value

    # close fist and keep it closed -> hold engages after the dwell and pins
    # depth at the PRE-gesture snapshot even though depth_norm jumped to 0.9
    for _ in range(14):  # ~0.47 s > dwell
        main._drive_from_hand(fake_signals(depth_norm=0.9, finger_curl_norm=0.9), dt, t, cfg, filters, pose, cs)
        t += dt
    check("grip engages depth hold after dwell", cs.depth_hold)
    check("depth held at pre-gesture value", abs(filters["depth"].value - depth_at_close) < 0.05,
          f"{filters['depth'].value:.3f} vs {depth_at_close:.3f}")

    # advance past grip_depth_hold_s while still closed -> hold releases (resume)
    for _ in range(int(cfg.filters.grip_depth_hold_s / dt) + 10):
        main._drive_from_hand(fake_signals(depth_norm=0.9, finger_curl_norm=0.9), dt, t, cfg, filters, pose, cs)
        t += dt
    check("depth resumes after grip timer", not cs.depth_hold)

    # roll tilt above +-10 deg (with dwell) -> holds depth & freezes height
    cs2 = main.ControlState(prev_pitch=(active.p1, active.p2, active.p3))
    filters2 = main._build_filters(cfg)
    for _ in range(5):
        main._drive_from_hand(fake_signals(height_norm=0.5), dt, t, cfg, filters2, pose, cs2)
        t += dt
    for _ in range(14):
        main._drive_from_hand(fake_signals(roll_tilt_delta=0.5, height_norm=0.8), dt, t, cfg, filters2, pose, cs2)
        t += dt
    check("roll activates depth+height hold", cs2.depth_hold and cs2.roll_active)
    check("height frozen during roll", abs(filters2["height"].value - 0.5) < 0.05,
          f"{filters2['height'].value:.3f}")
    # below-threshold tilt (~6 deg < 10 deg) must not activate roll hold
    cs3 = main.ControlState(prev_pitch=(active.p1, active.p2, active.p3))
    filters3 = main._build_filters(cfg)
    for _ in range(14):
        main._drive_from_hand(fake_signals(roll_tilt_delta=0.10), dt, t, cfg, filters3, pose, cs3)
        t += dt
    check("small tilt below 10 deg ignored", not cs3.roll_active)


def test_s_curve(cfg):
    print("SCurveLimiter (jerk-limited + target-velocity feedforward)")
    dt = 0.01
    j = cfg.motion.p1
    s = SCurveLimiter(j.rate_deg_s, j.accel_deg_s2, j.jerk_deg_s3, 0.0)

    # step response: gentle start, converges, no overshoot bounce
    first = s.update(60.0, dt)
    check("gentle start from standstill", first < 0.05, f"{first:.4f}")
    peak, prev = 0.0, first
    for _ in range(2000):
        prev = s.update(60.0, dt)
        peak = max(peak, prev)
    check("converges to step target", abs(prev - 60.0) < 0.5, f"{prev:.2f}")
    check("no overshoot on step", peak < 61.5, f"peak={peak:.2f}")

    # THE regression: chase a ramp target (like the 30 Hz stream) without the
    # accel-brake limit cycle a stop-distance trapezoid produces.
    s2 = SCurveLimiter(j.rate_deg_s, j.accel_deg_s2, j.jerk_deg_s3, 0.0)
    target, ramp_v = 0.0, 20.0
    for _ in range(150):  # 1.5 s settle
        target += ramp_v * dt
        s2.update(target, dt)
    vels = []
    for _ in range(100):  # 1 s steady chase
        target += ramp_v * dt
        s2.update(target, dt)
        vels.append(s2.velocity)
    band = max(vels) - min(vels)
    mean_v = sum(vels) / len(vels)
    check("tracks moving target near its speed", abs(mean_v - ramp_v) < 3.0, f"{mean_v:.1f}")
    check("NO accel-slow-accel limit cycle (vel band tight)", band < 5.0, f"band={band:.2f} deg/s")


def test_exp_smooth(cfg):
    print("ExpSmoothLimiter (first-order tracker)")
    dt = 0.01
    e = ExpSmoothLimiter(45.0, cfg.motion.exp_track_gain, 400.0, 0.0)
    peak = prev = 0.0
    for _ in range(1500):
        prev = e.update(50.0, dt)
        peak = max(peak, prev)
    check("converges", abs(prev - 50.0) < 0.5, f"{prev:.2f}")
    check("never overshoots", peak <= 50.0 + 1e-6, f"peak={peak:.3f}")


def test_min_jerk(cfg):
    print("min-jerk coordinated pose transition")
    check("s(0)=0, s(1)=1", min_jerk_s(0.0) == 0.0 and abs(min_jerk_s(1.0) - 1.0) < 1e-9)
    check("midpoint = 0.5", abs(min_jerk_s(0.5) - 0.5) < 1e-9)

    import main as m
    start = m.Pose(90.0, 98.1, 88.2, 90.0, 90.0, 1.0)   # home
    target = m.Pose(90.0, 45.0, 112.5, 90.0, 90.0, 1.0)  # active
    rate, dt = 6.0, 1 / 30.0
    tr = m.PoseTransition(start, target, rate, 130.5)
    pose = m.Pose(start.base, start.p1, start.p2, start.p3, start.roll, start.grip)

    prev_p1 = pose.p1
    first_done = tr.step(pose, dt)
    check("starts nearly at rest (min-jerk)", abs(pose.p1 - prev_p1) < 0.05 and not first_done)

    peak_v, steps = 0.0, 1
    prev_p1 = pose.p1
    done = False
    while not done and steps < 5000:
        done = tr.step(pose, dt)
        peak_v = max(peak_v, abs(pose.p1 - prev_p1) / dt)
        prev_p1 = pose.p1
        steps += 1
    check("arrives exactly on the target pose",
          done and abs(pose.p1 - 45.0) < 1e-6 and abs(pose.p2 - 112.5) < 1e-6)
    check("peak joint speed ~= configured rate", peak_v <= rate * 1.1, f"{peak_v:.2f}")
    # coordinated: p2 (smaller delta) finishes at the same moment as p1
    check("all joints arrive together (single duration)",
          abs(tr.duration - 1.875 * 53.1 / rate) < 1e-6, f"T={tr.duration:.2f}")


def test_motion_config(cfg):
    print("per-joint motion tuning defaults")
    m = cfg.motion
    check("p2 accel is the lowest (couples p1+p3)",
          m.p2.accel_deg_s2 < m.p1.accel_deg_s2 and m.p2.accel_deg_s2 < m.p3.accel_deg_s2)
    check("profile selectable", m.pitch_profile in ("trapezoid", "s_curve", "exp_smooth", "none"))
    import main as mn
    for prof in ("trapezoid", "s_curve", "exp_smooth", "none"):
        cfg.motion.pitch_profile = prof
        lim = mn._make_pitch_limiter(cfg, m.p1, 45.0)
        ok = (lim is None) if prof == "none" else (lim is not None)
        check(f"factory builds '{prof}'", ok)
    cfg.motion.pitch_profile = "s_curve"


# ============================================================================
# Firmware parity: a faithful Python port of the Zephyr firmware's V3 intake
# (src/servo_control.c + src/app_callbacks.c). If the host and firmware ever
# drift, these checks fail. Keep this table in sync with servo_control.c.
# ============================================================================
FW_MAGIC, FW_VERSION, FW_MASK, FW_LEN = 0xA6, 0x03, 0x3F, 19
FW_MIN = [0.0, 0.0, 45.0, 0.0, 0.0, 36.0]
FW_MAX = [180.0, 100.0, 180.0, 180.0, 180.0, 166.5]
FW_HOME = [90.0, 98.1, 88.2, 90.0, 90.0, 166.5]


def fw_apply(buf: bytes, targets: list[float]) -> tuple[bool, list[float]]:
    """Mirror servo_control_apply_packet(): returns (accepted, new_targets)."""
    t = list(targets)
    if len(buf) != FW_LEN:
        return False, t
    if buf[0] != FW_MAGIC or buf[1] != FW_VERSION:
        return False, t
    mode = chr(buf[4])
    if mode not in ("A", "H", "M"):
        return False, t
    if buf[5] != FW_MASK:
        return False, t
    checksum = 0
    for i in range(FW_LEN - 1):
        checksum ^= buf[i]
    if checksum != buf[FW_LEN - 1]:
        return False, t
    if mode == "A":
        for i in range(6):
            q = (buf[6 + 2 * i] << 8) | buf[7 + 2 * i]
            angle = (q / 65535.0) * 180.0
            angle = max(FW_MIN[i], min(FW_MAX[i], angle))
            if abs(angle - t[i]) >= 0.2:        # TARGET_DEADBAND_DEG
                t[i] = angle
        return True, t
    if mode == "M":
        return True, list(FW_HOME)
    return True, t  # 'H' holds


def test_firmware_parity(cfg):
    print("host <-> firmware V3 parity (faithful port of servo_control.c)")
    # A normal active packet is accepted and decodes to the sent angles.
    pkt = protocol.build_packet(1, "A", 90, 20, 120, 150, 70, 100)
    ok, t = fw_apply(pkt, [0.0] * 6)
    check("valid A accepted", ok)
    ref = protocol.parse_packet(pkt)
    check("firmware angles match protocol.parse", all(
        abs(t[i] - v) < 0.05 for i, v in enumerate(
            [ref["base"], ref["p1"], ref["p2"], ref["p3"], ref["roll"], ref["gripper"]])), str(t))

    # Per-joint clamps match the contract table.
    ext = protocol.build_packet(2, "A", 200, 200, 0, 200, 200, 200)  # all over/under range
    _, tc = fw_apply(ext, [0.0] * 6)
    check("clamps: base<=180 p1<=100 p2>=45 grip<=166.5",
          tc[0] == 180.0 and tc[1] == 100.0 and tc[2] == 45.0 and abs(tc[5] - 166.5) < 0.05, str(tc))

    # Mode M snaps to the exact home pose; mode H holds the previous targets.
    _, th = fw_apply(protocol.build_packet(3, "M", 0, 0, 0, 0, 0, 0), [10.0] * 6)
    check("M -> exact home pose", all(abs(th[i] - FW_HOME[i]) < 1e-6 for i in range(6)), str(th))
    held = [11.0, 22.0, 90.0, 33.0, 44.0, 100.0]
    _, thold = fw_apply(protocol.build_packet(4, "H", 90, 90, 90, 90, 90, 90), list(held))
    check("H holds previous targets", thold == held, str(thold))

    # Rejections: bad checksum, wrong version (V2), removed mode 'S', bad length, bad mask.
    bad = bytearray(pkt); bad[-1] ^= 0xFF
    check("rejects bad checksum", not fw_apply(bytes(bad), [0.0] * 6)[0])
    v2 = bytearray(pkt); v2[1] = 0x02
    check("rejects V2 version", not fw_apply(bytes(v2), [0.0] * 6)[0])
    s_mode = bytearray(pkt); s_mode[4] = ord("S")
    check("rejects mode S (removed in V3)", not fw_apply(bytes(s_mode), [0.0] * 6)[0])
    check("rejects wrong length", not fw_apply(pkt[:-1], [0.0] * 6)[0])
    bad_mask = bytearray(pkt); bad_mask[5] = 0x1F
    check("rejects wrong mask", not fw_apply(bytes(bad_mask), [0.0] * 6)[0])

    # The actual host poses (home + active) survive the round trip unclamped.
    import main as m
    for name, pc in (("home", cfg.home_pose), ("active", cfg.active_pose)):
        pose = m._pose_from_cfg(pc)
        cmd = pose.command("A", cfg.gripper)
        cmd.sequence = 5
        _, tp = fw_apply(cmd.to_packet(), [0.0] * 6)
        want = [pc.base_deg, pc.p1_deg, pc.p2_deg, pc.p3_deg, pc.roll_deg, cmd.gripper_deg]
        check(f"{name} pose within firmware limits (unclamped)",
              all(abs(tp[i] - want[i]) < 0.05 for i in range(6)), f"{tp} vs {want}")


def test_no_stray_modes():
    print("host never emits a mode outside {A,H,M}")
    import re
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    # every  mode="X"  or  .command("X"  literal used in the host loop
    modes = set(re.findall(r'command\(\s*["\']([A-Z])["\']', src))
    modes |= set(re.findall(r'mode\s*[,=]\s*["\']([A-Z])["\']', src))
    check("only A/H/M used", modes.issubset({"A", "H", "M"}), str(sorted(modes)))


def test_transport_nonblocking(cfg):
    print("transport: a down/booting ESP32 must NOT stall the camera loop")
    import time as _t
    import main as m
    from config import TcpConfig, WebSocketConfig
    from tcp_comm import TcpController
    from websocket_comm import WebSocketController

    cmd = m.Pose(90.0, 45.0, 90.0, 90.0, 90.0, 1.0).command("A", cfg.gripper)
    # unroutable TEST-NET-ish address: connect() hangs until timeout — which
    # must now happen on a background thread, never on the caller
    wcfg = WebSocketConfig(enabled=True, dry_run=False, host="10.255.255.1",
                           port=4210, reconnect_interval_s=0.05, write_hz=1000.0)
    tcfg = TcpConfig(enabled=True, dry_run=False, host="10.255.255.1",
                     port=4210, reconnect_interval_s=0.05, write_hz=1000.0)
    for name, ctrl in (("websocket", WebSocketController(wcfg)),
                       ("tcp", TcpController(tcfg))):
        t0 = _t.perf_counter()
        sent = False
        for _ in range(5):
            sent = ctrl.send(cmd) or sent
        elapsed = _t.perf_counter() - t0
        check(f"{name}: send() with down host returns instantly (bg connect)",
              elapsed < 0.25 and not sent, f"{elapsed * 1000:.0f} ms")


def test_transport_pump(cfg):
    print("transport pump: sender thread streams even while the camera loop stalls")
    import time as _t
    import main as m
    from config import WebSocketConfig
    from websocket_comm import WebSocketController

    ctrl = WebSocketController(WebSocketConfig(enabled=True, dry_run=True, write_hz=100.0))
    pump = m.TransportPump(ctrl, 100.0)
    cmd = m.Pose(90.0, 45.0, 90.0, 90.0, 90.0, 1.0).command("A", cfg.gripper)
    t0 = _t.perf_counter()
    pump.update(cmd)
    handoff = _t.perf_counter() - t0
    _t.sleep(0.25)                    # simulate a stalled vision loop
    sent = ctrl._sequence             # dry-run sends still advance the sequence
    pump.stop()
    check("update() hands off instantly (no socket work on caller)",
          handoff < 0.005, f"{handoff * 1000:.2f} ms")
    check("pump kept streaming during a 250 ms loop stall (keepalive)",
          sent >= 10, f"{sent} packets in 250 ms")


def main() -> int:
    cfg = load_config("config/calibration.json")
    test_protocol()
    test_integrator(cfg)
    test_one_euro()
    test_accel_slew(cfg)
    test_s_curve(cfg)
    test_exp_smooth(cfg)
    test_min_jerk(cfg)
    test_motion_config(cfg)
    test_gripper(cfg)
    test_direct_pitch(cfg)
    test_depth_response(cfg)
    test_output_clamp(cfg)
    test_two_hand(cfg)
    test_roll_depth_decouple(cfg)
    test_ik(cfg)
    test_ramp(cfg)
    test_depth_hold(cfg)
    test_firmware_parity(cfg)
    test_transport_nonblocking(cfg)
    test_transport_pump(cfg)
    test_no_stray_modes()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
