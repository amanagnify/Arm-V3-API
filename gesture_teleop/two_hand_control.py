from __future__ import annotations

"""Two-hand split-screen control scheme.

Screen halves assign roles (mirrored view, so it matches the user's hands):
  RIGHT half hand -> base yaw (x-deflection), p1 (depth = hand size), p2 (height)
  LEFT  half hand -> base yaw too, p3 (height), roll (tilt), gripper (curl)

Neutral / anchor points move ONLY when the R or L key is pressed (the L/R
Agnisys-logo markers). All motion is relative to that fixed anchor, so the
markers never drift while you work.

Right-hand FIST (dwell) PAUSES p1/p2 (and the right hand's yaw contribution).
To RESUME you must OPEN -> CLOSE -> OPEN the right fist (a deliberate double
gesture, so opening the hand to relax never resumes by accident).

RIGHT peace / V sign (index+middle up, ring+pinky curled) = go home.

Everything here outputs TARGETS; smoothing/clamping stays in main.py's
per-joint limiters and the firmware profile. `None` targets mean HOLD.
"""

import math
from dataclasses import dataclass

from config import AppConfig
from filters import apply_signed_deadband, clamp, make_input_filter
from hand_mapping import _curl_to_gripper
from vision import HandObservation

ROLE_LEFT = "L"
ROLE_RIGHT = "R"


def curl_norm(obs: HandObservation, cfg: AppConfig) -> float:
    """Normalized finger curl in [0,1] (same formula as extract_signals)."""
    span = max(cfg.vision.grip_closed_reference - cfg.vision.grip_open_reference, 1e-6)
    return clamp((obs.finger_curl_metric - cfg.vision.grip_open_reference) / span, 0.0, 1.0)


def is_peace_sign(obs: HandObservation) -> bool:
    """Index + middle extended, ring + pinky curled (V / peace sign).
    Landmark y grows DOWNWARD in image coordinates."""
    lm = obs.normalized_landmarks
    index_up = lm[8, 1] < lm[6, 1] - 0.02
    middle_up = lm[12, 1] < lm[10, 1] - 0.02
    ring_curl = lm[16, 1] > lm[14, 1]
    pinky_curl = lm[20, 1] > lm[18, 1]
    return bool(index_up and middle_up and ring_curl and pinky_curl)


@dataclass
class HandChannelState:
    anchor_x: float = 0.5                  # yaw zero (both) / re-set only by key
    anchor_y: float = 0.5                  # p2 (right) / p3 (left) height zero
    anchor_size: float = 0.10              # right: hand-size (depth) reference
    anchor_tilt: float = 0.0               # left: roll tilt reference
    base_p1: float = 45.0                  # pose values captured at key press
    base_p2: float = 112.5
    base_p3: float = 90.0
    locked: bool = False                   # right only: p1/p2 paused
    curl_above_since: float | None = None  # right lock dwell
    resume_stage: int = 0                  # right resume: 0 idle,1 saw-open,2 saw-close
    last_seen: float = -1e9                # for input-filter seeding + dropout coast
    last_p1: float | None = None           # last computed targets, kept alive
    last_p2: float | None = None           # through short tracking dropouts
    last_p3: float | None = None


@dataclass
class TwoHandFrame:
    """Per-frame output. None target = hold the current value."""
    yaw_input: float = 0.0                 # combined, deadbanded, [-1,1]
    p1: float | None = None
    p2: float | None = None
    p3: float | None = None
    roll_input: float = 0.0
    grip: float | None = None
    right_present: bool = False
    left_present: bool = False
    right_engaged: bool = False            # following (present & not locked)
    left_engaged: bool = False
    right_locked: bool = False
    home_requested: bool = False
    right_obs: HandObservation | None = None
    left_obs: HandObservation | None = None
    right_anchor: tuple[float, float] = (0.73, 0.55)
    left_anchor: tuple[float, float] = (0.27, 0.55)
    right_deflection: float = 0.0
    left_deflection: float = 0.0
    r_height: float = 0.5                   # right-hand height (p2), 0.5 = neutral
    l_height: float = 0.5                   # left-hand height (p3), 0.5 = neutral


class TwoHandController:
    def __init__(self, config: AppConfig) -> None:
        v = config.vision
        th = config.two_hand
        fcfg = config.filters
        self.right = HandChannelState(anchor_x=v.neutral_right_x, anchor_y=v.neutral_right_y,
                                      base_p1=th.neutral_p1, base_p2=th.neutral_p2)
        self.left = HandChannelState(anchor_x=v.neutral_left_x, anchor_y=v.neutral_left_y,
                                     base_p3=th.neutral_p3)
        # One-Euro input filters per role. Raw MediaPipe positions/sizes are
        # noisy; unfiltered they turn into velocity-feedforward spikes in the
        # S-curve limiters (stop-and-jump motion). Curl stays RAW so the
        # lock/resume and grip gestures keep their crisp timing.
        self._rf = {"x": make_input_filter(config), "y": make_input_filter(config),
                    "size": make_input_filter(config, min_cutoff=fcfg.depth_min_cutoff,
                                              beta=fcfg.depth_beta)}
        self._lf = {"x": make_input_filter(config), "y": make_input_filter(config),
                    "tilt": make_input_filter(config)}
        self._last_role_right: bool | None = None
        self._home_since: float | None = None
        self._home_block_until: float = 0.0

    def _filtered(self, filt: dict, state: HandChannelState, now: float,
                  grace: float, raw: dict) -> dict:
        """Filter one hand's raw signals. On (re)appearance after a gap the
        filters seed from the raw values, so a returning hand never slews in
        from a stale position."""
        gap = now - state.last_seen
        if gap > grace:
            for k, f in filt.items():
                f.reset(raw[k])
            return dict(raw)
        fdt = clamp(gap, 1.0 / 120.0, 0.1)
        return {k: f.update(raw[k], fdt) for k, f in filt.items()}

    # ------------------------------------------------------------ roles
    def assign_roles(self, hands: list[HandObservation], cfg: AppConfig
                     ) -> tuple[HandObservation | None, HandObservation | None]:
        """(left_obs, right_obs). Two+ hands: leftmost=LEFT, rightmost=RIGHT.
        One hand: by screen half with hysteresis near the midline."""
        if not hands:
            self._last_role_right = None
            return None, None
        if len(hands) >= 2:
            ordered = sorted(hands, key=lambda o: o.center_xy[0])
            self._last_role_right = None
            return ordered[0], ordered[-1]

        obs = hands[0]
        x = obs.center_xy[0]
        hyst = cfg.two_hand.role_hysteresis
        if self._last_role_right is True:
            is_right = x > 0.5 - hyst
        elif self._last_role_right is False:
            is_right = x >= 0.5 + hyst
        else:
            is_right = x >= 0.5
        self._last_role_right = is_right
        return (None, obs) if is_right else (obs, None)

    # ------------------------------------------------------- anchoring
    def set_anchors(self, left_obs: HandObservation | None,
                    right_obs: HandObservation | None, pose, cfg: AppConfig) -> None:
        """R/L key: the ONLY thing that moves a neutral point. Re-centers the
        anchor to the live hand AND drives the arm to the neutral pose so each
        hand's neutral maps to a known posture (right -> p1/p2, left -> p3/roll,
        gripper opens with an open left palm). The arm glides there smoothly."""
        th = cfg.two_hand
        if right_obs is not None:
            rs = self.right
            rs.anchor_x, rs.anchor_y = right_obs.center_xy
            rs.anchor_size = max(right_obs.depth_metric, 1e-6)
            # seed the input filters at the anchor so the key press lands
            # exactly on the neutral pose (no filter-lag offset)
            self._rf["x"].reset(rs.anchor_x)
            self._rf["y"].reset(rs.anchor_y)
            self._rf["size"].reset(rs.anchor_size)
            rs.base_p1, rs.base_p2 = th.neutral_p1, th.neutral_p2
            rs.locked = False
            rs.curl_above_since = None
            rs.resume_stage = 0
            rs.last_p1 = rs.last_p2 = None  # old coast targets are stale now
        if left_obs is not None:
            ls = self.left
            ls.anchor_x, ls.anchor_y = left_obs.center_xy
            ls.anchor_tilt = left_obs.wrist_tilt_metric
            self._lf["x"].reset(ls.anchor_x)
            self._lf["y"].reset(ls.anchor_y)
            self._lf["tilt"].reset(ls.anchor_tilt)
            ls.base_p3 = th.neutral_p3
            ls.last_p3 = None
            pose.roll = th.neutral_roll   # roll is rate-integrated; snap its neutral

    # ------------------------------------------------------ main update
    def update(self, hands: list[HandObservation], pose, now: float,
               cfg: AppConfig) -> TwoHandFrame:
        th = cfg.two_hand
        left_obs, right_obs = self.assign_roles(hands, cfg)
        frame = TwoHandFrame(right_obs=right_obs, left_obs=left_obs,
                             right_present=right_obs is not None,
                             left_present=left_obs is not None)
        yaw_right = 0.0
        yaw_left = 0.0

        # ---------------- RIGHT: yaw + p1(depth) + p2(height); fist pauses
        rs = self.right
        if right_obs is not None:
            sig = self._filtered(self._rf, rs, now, th.track_grace_s, {
                "x": right_obs.center_xy[0], "y": right_obs.center_xy[1],
                "size": max(right_obs.depth_metric, 1e-6)})
            rs.last_seen = now
            curl = curl_norm(right_obs, cfg)
            if not rs.locked:
                if curl >= th.lock_curl:
                    rs.curl_above_since = rs.curl_above_since or now
                    if (now - rs.curl_above_since) >= th.lock_dwell_s:
                        rs.locked = True
                        rs.resume_stage = 0
                else:
                    rs.curl_above_since = None
            else:
                # resume only on OPEN -> CLOSE -> OPEN
                if rs.resume_stage == 0 and curl <= th.unlock_curl:
                    rs.resume_stage = 1
                elif rs.resume_stage == 1 and curl >= th.lock_curl:
                    rs.resume_stage = 2
                elif rs.resume_stage == 2 and curl <= th.unlock_curl:
                    rs.locked = False
                    rs.resume_stage = 0
                    rs.curl_above_since = None

            if not rs.locked:
                x, y = sig["x"], sig["y"]
                yaw_right = clamp((x - rs.anchor_x) / max(th.yaw_deflection_span, 1e-6),
                                  -1.0, 1.0)
                # p1: symmetric log-size depth relative to the anchor size
                ratio = sig["size"] / rs.anchor_size
                span1 = cfg.limits.p1_max - cfg.limits.p1_min
                d_p1 = span1 * cfg.mapping.depth_gain * math.log(max(ratio, 1e-3))
                if cfg.mapping.lower_depth_invert:
                    d_p1 = -d_p1
                frame.p1 = clamp(rs.base_p1 + d_p1, cfg.limits.p1_min, cfg.limits.p1_max)
                # p2: hand up -> p2 joint up (servo down when flipped)
                up = (rs.anchor_y - y) / 0.5
                d_servo = (-up if cfg.mapping.p2_invert else up) * th.p2_height_gain_deg
                if cfg.mapping.p2_invert:
                    p2_lo, p2_hi = 180.0 - cfg.limits.p2_max, 180.0 - cfg.limits.p2_min
                else:
                    p2_lo, p2_hi = cfg.limits.p2_min, cfg.limits.p2_max
                frame.p2 = clamp(rs.base_p2 + d_servo, p2_lo, p2_hi)
                rs.last_p1, rs.last_p2 = frame.p1, frame.p2
                frame.r_height = clamp(0.5 + 0.5 * up, 0.0, 1.0)
                frame.right_deflection = clamp(
                    math.hypot(x - rs.anchor_x, y - rs.anchor_y) / 0.3, 0.0, 1.0)
        else:
            # hand gone: KEEP the lock/anchor. Require a fresh OPEN->CLOSE->OPEN
            # when it returns, so leaving view never resumes on its own.
            rs.curl_above_since = None
            if rs.locked:
                rs.resume_stage = 0
            elif (now - rs.last_seen) <= th.track_grace_s:
                # coast: a one-frame MediaPipe flicker keeps the last targets
                # alive so the limiters never zero their velocity mid-move
                frame.p1, frame.p2 = rs.last_p1, rs.last_p2

        # ---------------- LEFT: yaw + p3(height) + roll(tilt) + grip(curl)
        ls = self.left
        if left_obs is not None:
            sig = self._filtered(self._lf, ls, now, th.track_grace_s, {
                "x": left_obs.center_xy[0], "y": left_obs.center_xy[1],
                "tilt": left_obs.wrist_tilt_metric})
            ls.last_seen = now
            x, y = sig["x"], sig["y"]
            curl = curl_norm(left_obs, cfg)
            yaw_left = clamp((x - ls.anchor_x) / max(th.yaw_deflection_span, 1e-6),
                             -1.0, 1.0)
            up = (ls.anchor_y - y) / 0.5
            frame.p3 = clamp(ls.base_p3 + up * th.p3_height_gain_deg,
                             cfg.limits.p3_min, cfg.limits.p3_max)
            ls.last_p3 = frame.p3
            frame.l_height = clamp(0.5 + 0.5 * up, 0.0, 1.0)
            tilt_delta = sig["tilt"] - ls.anchor_tilt
            frame.roll_input = clamp(
                tilt_delta / max(cfg.mapping.roll_tilt_span_rad, 1e-6), -1.0, 1.0)
            frame.grip = _curl_to_gripper(curl, cfg)  # natural: open->open, fist->closed
            frame.left_deflection = clamp(
                math.hypot(x - ls.anchor_x, y - ls.anchor_y) / 0.3, 0.0, 1.0)
        elif (now - ls.last_seen) <= th.track_grace_s:
            frame.p3 = ls.last_p3  # coast through tracking flickers

        # ---------------- combined yaw (each hand deadbanded independently)
        db = cfg.mapping.yaw_deadband
        frame.yaw_input = clamp(apply_signed_deadband(yaw_right, db)
                                + apply_signed_deadband(yaw_left, db), -1.0, 1.0)

        # ---------------- RIGHT peace / V sign -> go home
        peace = right_obs is not None and is_peace_sign(right_obs)
        if peace and now >= self._home_block_until:
            self._home_since = self._home_since or now
            if (now - self._home_since) >= th.home_gesture_dwell_s:
                frame.home_requested = True
                self._home_block_until = now + th.home_gesture_refractory_s
                self._home_since = None
        else:
            self._home_since = None

        frame.right_engaged = right_obs is not None and not rs.locked
        frame.left_engaged = left_obs is not None
        frame.right_locked = rs.locked
        frame.right_anchor = (rs.anchor_x, rs.anchor_y)
        frame.left_anchor = (ls.anchor_x, ls.anchor_y)
        return frame
