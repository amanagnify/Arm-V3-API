#!/usr/bin/env python3
"""PS3 / generic gamepad controller for the Arm-V3 6-DOF robot.

Integration contract
--------------------
Call ``probe_joystick()`` once at startup.  It returns a ``JoystickDriver``
when a joystick is found, or ``None`` when none is connected.  The caller
falls back to gesture mode when ``None`` is returned.

Inside the main loop call ``JoystickDriver.read(dt)`` every frame.  It
returns a ``JoystickState`` dataclass that carries:

  * ``base_deg``   - integrated yaw in servo-space [0, 180]
  * ``p1_deg``     - integrated lower-pitch [0, p1_max]
  * ``p2_deg``     - integrated middle-pitch (raw joint; firmware inverts)
  * ``p3_deg``     - integrated upper-pitch [0, p3_max]
  * ``roll_deg``   - integrated wrist/roll [0, 180]
  * ``grip_open``  - gripper open fraction [0.0, 1.0]
  * ``estop``      - True if SELECT was pressed
  * ``freeze``     - True if START was pressed (toggle handled outside)
  * ``go_home``    - True if PS/Guide button pressed

PS3 axis layout (ds4drv / SDL2 mapping on Linux)
-------------------------------------------------
  Axis 0  : Left-X   → base yaw
  Axis 1  : Left-Y   → p1 lower (push up = extend)
  Axis 2  : L2 trigger  (-1 fully released → +1 fully pressed)
  Axis 3  : Right-X  → roll / wrist
  Axis 4  : Right-Y  → p3 upper pitch
  Axis 5  : R2 trigger  (-1 released → +1 pressed)

  Button 0  : Cross (X)   - gripper close (hold)
  Button 3  : Square      - gripper open  (hold)
  Button 6  : L2 button   (digital shadow)
  Button 7  : R2 button   (digital shadow)
  Button 8  : SELECT      - e-stop toggle
  Button 9  : START       - freeze toggle
  Button 10 : L3          - reset all joints to home
  Button 11 : R3          - (unused, reserved)
  Button 12 : PS/Guide    - go home

  Hat 0     : D-Pad       - D-Left/Right trims roll (±2 deg per press)

Wrist/roll is driven by Right-X (Axis 3) in velocity-integration mode
(same feel as yaw) OR by the D-Pad left/right buttons for single-step trim.
Both inputs combine additively so either can be used independently.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig

# pygame is optional; the rest of the project doesn't require it so we do a
# deferred import and surface a clean error when the package is absent.
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False



# ---------------------------------------------------------------------------
# Default axis / button indices – override at construction time if your
# controller reports a different mapping.
# ---------------------------------------------------------------------------
_AXIS_LX      = 0   # left-X  → base yaw
_AXIS_LY      = 1   # left-Y  → p1 lower (−1 = up = extend)
_AXIS_L2      = 2   # L2 trigger
_AXIS_RX      = 3   # right-X → wrist / roll
_AXIS_RY      = 4   # right-Y → p3 upper
_AXIS_R2      = 5   # R2 trigger

_BTN_CROSS    = 0   # X / Cross  – gripper close
_BTN_CIRCLE   = 1   # Circle
_BTN_SQUARE   = 3   # Square     – gripper open
_BTN_L1       = 4
_BTN_R1       = 5
_BTN_L2       = 6
_BTN_R2       = 7
_BTN_SELECT   = 8   # e-stop toggle
_BTN_START    = 9   # freeze toggle
_BTN_L3       = 10  # reset to home
_BTN_R3       = 11
_BTN_PS       = 12  # go-home command

# D-Pad roll trim step
_DPAD_ROLL_STEP_DEG = 3.0

# Deadband applied to all analogue axes
_DEADBAND = 0.08



def _deadband(v: float, db: float = _DEADBAND) -> float:
    return 0.0 if abs(v) < db else v


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _trigger_01(raw: float) -> float:
    """Convert a trigger axis (-1 released, +1 pressed) to [0, 1]."""
    return _clamp((raw + 1.0) / 2.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public dataclass returned by JoystickDriver.read()
# ---------------------------------------------------------------------------
@dataclass
class JoystickState:
    """Absolute joint targets derived from one joystick frame."""
    base_deg:  float = 90.0   # yaw  [0, 180]
    p1_deg:    float = 45.0   # lower pitch [p1_min, p1_max]
    p2_deg:    float = 112.5  # middle pitch (joint-space, pre-inversion) [0, 135]
    p3_deg:    float = 90.0   # upper pitch [0, 180]
    roll_deg:  float = 90.0   # wrist/roll  [0, 180]
    grip_open: float = 1.0    # gripper open fraction [0, 1]

    # Event flags (pulse once per press; caller handles toggle logic)
    estop:     bool = False
    freeze:    bool = False
    go_home:   bool = False
    reset:     bool = False   # L3 → snap back to home values


# ---------------------------------------------------------------------------
# Joystick driver
# ---------------------------------------------------------------------------
class JoystickDriver:
    """Wraps a single pygame Joystick and integrates incremental commands.

    Joint positions accumulate from neutral each time :meth:`read` is called
    so the arm drifts smoothly in whichever direction the stick points.
    Releasing a stick stops the motion (velocity-mode, not position-snap).
    """

    def __init__(
        self,
        js: "pygame.joystick.Joystick",
        config: Optional[AppConfig] = None,
        *,
        # Seed values: the arm's current pose at engagement time
        seed_base: float  = 90.0,
        seed_p1:   float  = 45.0,
        seed_p2:   float  = 112.5,
        seed_p3:   float  = 90.0,
        seed_roll: float  = 90.0,
        seed_grip: float  = 1.0,
    ) -> None:
        self._js = js
        
        if config is None:
            try:
                from config import AppConfig as _AppConfig
                config = _AppConfig()
            except ImportError:
                pass
        self._config = config

        self._limits = {}
        self._base_rate = 25.0
        self._p1_rate = 30.0
        self._p3_rate = 35.0
        self._roll_rate = 40.0
        self._p2_rate = 25.0
        self._grip_rate = 1.5
        self._roll_exp = 1.6
        self._yaw_exp = 1.6

        # Sync rates, limits, exponents from config
        self.set_rates_from_config(config)

        # Integrated positions
        blo, bhi = self._limits.get("base", (0.0, 180.0))
        plo, phi = self._limits.get("p1", (0.0, 90.0))
        p2lo, p2hi = self._limits.get("p2", (0.0, 135.0))
        p3lo, p3hi = self._limits.get("p3", (0.0, 180.0))
        rlo, rhi = self._limits.get("roll", (0.0, 180.0))

        self._base = _clamp(seed_base, blo, bhi)
        self._p1   = _clamp(seed_p1,   plo, phi)

        # p2 seed is passed in the configured servo space; convert if inverted.
        p2_joint = 180.0 - seed_p2 if (config and config.mapping.p2_invert) else seed_p2
        self._p2   = _clamp(p2_joint,   p2lo, p2hi)

        self._p3   = _clamp(seed_p3,   p3lo, p3hi)
        self._roll = _clamp(seed_roll, rlo, rhi)
        self._grip = _clamp(seed_grip, 0.0, 1.0)

        # Button edge-tracking for event flags
        self._prev_btns: dict[int, bool] = {}
        self._prev_hat: tuple[int, int] = (0, 0)

        self._name = js.get_name()

    @property
    def name(self) -> str:
        return self._name

    def reseed(self, base: float, p1: float, p2: float, p3: float,
               roll: float, grip: float) -> None:
        """Sync internal state to the arm's current pose (call on engagement)."""
        lo, hi = self._limits.get("base", (0.0, 180.0));  self._base = _clamp(base, lo, hi)
        lo, hi = self._limits.get("p1", (0.0, 90.0));    self._p1   = _clamp(p1,   lo, hi)
        
        # p2 is passed in the configured servo space; convert if inverted.
        p2_joint = 180.0 - p2 if (self._config and self._config.mapping.p2_invert) else p2
        lo, hi = self._limits.get("p2", (0.0, 135.0));    self._p2   = _clamp(p2_joint, lo, hi)
        
        lo, hi = self._limits.get("p3", (0.0, 180.0));    self._p3   = _clamp(p3,   lo, hi)
        lo, hi = self._limits.get("roll", (0.0, 180.0));  self._roll = _clamp(roll, lo, hi)
        self._grip = _clamp(grip, 0.0, 1.0)

    def set_rates_from_config(self, config) -> None:
        """Sync motion rates and limits from AppConfig so joystick feel matches gesture mode."""
        if config is not None:
            self._config = config
        
        if self._config is None:
            return

        try:
            cfg = self._config
            # Limits
            lim = cfg.limits
            self._limits = dict(
                base=(lim.base_min, lim.base_max),
                p1=(lim.p1_min, lim.p1_max),
                p2=(lim.p2_min, lim.p2_max),
                p3=(lim.p3_min, lim.p3_max),
                roll=(lim.roll_min, lim.roll_max),
            )
            # Rates
            self._base_rate = cfg.mapping.yaw_rate_deg_s
            self._roll_rate = cfg.mapping.roll_rate_deg_s
            self._p1_rate   = cfg.motion.p1.rate_deg_s
            self._p2_rate   = cfg.motion.p2.rate_deg_s
            self._p3_rate   = cfg.motion.p3.rate_deg_s
            self._grip_rate = cfg.filters.gripper_rate_per_s
            
            # Exponents
            self._yaw_exp   = cfg.mapping.yaw_exponent
            self._roll_exp  = cfg.mapping.roll_exponent
        except AttributeError:
            pass  # keep defaults if config structure differs

    def _axis(self, idx: int) -> float:
        """Safe axis read: returns 0 when the controller has fewer axes."""
        js = self._js
        return js.get_axis(idx) if js.get_numaxes() > idx else 0.0

    def _btn(self, idx: int) -> bool:
        js = self._js
        return bool(js.get_button(idx)) if js.get_numbuttons() > idx else False

    def _hat(self, idx: int = 0) -> tuple[int, int]:
        js = self._js
        return js.get_hat(idx) if js.get_numhats() > idx else (0, 0)

    def _rising_edge(self, btn_idx: int, cur: bool) -> bool:
        """True only on the first frame the button is pressed."""
        prev = self._prev_btns.get(btn_idx, False)
        self._prev_btns[btn_idx] = cur
        return cur and not prev

    def read(self, dt: float) -> JoystickState:
        """Consume one frame of joystick data and return the integrated state.

        Parameters
        ----------
        dt:
            Time elapsed since the last call (seconds).  Use the same dt as
            the camera loop so integration speed matches configured rates.
        """
        dt = max(0.001, min(dt, 0.1))  # guard against stalls

        pygame.event.pump()  # flush the OS event queue

        # ---- raw axis reads -----------------------------------------------
        lx  = _deadband(self._axis(_AXIS_LX))   # base yaw (left stick left/right)
        ly  = _deadband(self._axis(_AXIS_LY))   # p1 lower  (−1 = up = left stick up/down)
        ry  = _deadband(self._axis(_AXIS_RY))   # p3 upper  (−1 = up = right stick up/down)

        # ---- non-linear shaping (same feel as gesture yaw) ----------------
        lx_shaped = math.copysign(abs(lx) ** self._yaw_exp, lx)

        # ---- integrate joints (velocity-mode) -----------------------------
        blo, bhi = self._limits.get("base", (0.0, 180.0))
        self._base = _clamp(
            self._base + lx_shaped * self._base_rate * dt, blo, bhi
        )

        # Left-Y up (ly < 0) → extend p1 (increase)
        plo, phi = self._limits.get("p1", (0.0, 90.0))
        self._p1 = _clamp(
            self._p1 + (-ly) * self._p1_rate * dt, plo, phi
        )

        # p3 upper: right-Y up (ry < 0) → increase p3
        p3lo, p3hi = self._limits.get("p3", (0.0, 180.0))
        self._p3 = _clamp(
            self._p3 + (-ry) * self._p3_rate * dt, p3lo, p3hi
        )

        # Wrist/Roll: L1 rolls left (-), R1 rolls right (+)
        roll_dir = 0.0
        if self._btn(_BTN_R1):
            roll_dir += 1.0
        if self._btn(_BTN_L1):
            roll_dir -= 1.0

        rlo, rhi = self._limits.get("roll", (0.0, 180.0))
        self._roll = _clamp(
            self._roll + roll_dir * self._roll_rate * dt, rlo, rhi
        )

        # p2 middle: follow average of p1 and p3 (coupled joint, pre-inversion)
        # The firmware inverts p2; here we just track the geometric average
        # so the wrist stays level unless p3 is deliberately angled.
        p2lo, p2hi = self._limits.get("p2", (0.0, 135.0))
        p2_target = (self._p1 + self._p3) / 2.0
        # Clamp into the joint-space limits (pre-flip)
        self._p2 = _clamp(p2_target, p2lo, p2hi)

        # ---- gripper: L2 closes, R2 opens ---------------------------------
        # Support both analog triggers and digital buttons
        l2_pressed = self._btn(_BTN_L2) or (_trigger_01(self._axis(_AXIS_L2)) > 0.1)
        r2_pressed = self._btn(_BTN_R2) or (_trigger_01(self._axis(_AXIS_R2)) > 0.1)
        
        # Support cross and square as backups
        if self._btn(_BTN_CROSS):
            l2_pressed = True
        elif self._btn(_BTN_SQUARE):
            r2_pressed = True

        grip_dir = 0.0
        if r2_pressed:
            grip_dir += 1.0
        if l2_pressed:
            grip_dir -= 1.0

        self._grip = _clamp(self._grip + grip_dir * self._grip_rate * dt, 0.0, 1.0)

        # ---- D-Pad roll trim (single-step per press) ----------------------
        hat = self._hat()
        hat_x = hat[0]  # −1 left, +1 right
        prev_hat_x = self._prev_hat[0]
        if hat_x != 0 and hat_x != prev_hat_x:
            self._roll = _clamp(
                self._roll + hat_x * _DPAD_ROLL_STEP_DEG, rlo, rhi
            )
        self._prev_hat = hat

        # ---- event flags (rising edge only) ------------------------------
        sel_pressed = self._rising_edge(_BTN_SELECT, self._btn(_BTN_SELECT))
        sta_pressed = self._rising_edge(_BTN_START,  self._btn(_BTN_START))
        ps_pressed  = self._rising_edge(_BTN_PS,     self._btn(_BTN_PS))
        rst_pressed = self._rising_edge(_BTN_L3,     self._btn(_BTN_L3))

        return JoystickState(
            base_deg  = round(self._base, 1),
            p1_deg    = round(self._p1,   1),
            p2_deg    = round(self._p2,   1),
            p3_deg    = round(self._p3,   1),
            roll_deg  = round(self._roll, 1),
            grip_open = round(self._grip, 3),
            estop     = sel_pressed,
            freeze    = sta_pressed,
            go_home   = ps_pressed,
            reset     = rst_pressed,
        )


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------
def probe_joystick(
    *,
    index: int = 0,
    verbose: bool = True,
    seed_pose: Optional[dict] = None,
    config: Optional[AppConfig] = None,
) -> Optional[JoystickDriver]:
    """Try to initialise pygame and open a joystick.

    Returns a :class:`JoystickDriver` on success, or ``None`` when:

    * pygame is not installed  →  falls back to gesture mode
    * No joystick is connected →  falls back to gesture mode

    Parameters
    ----------
    index:
        Joystick index to open (0 = first connected).
    verbose:
        Print status messages to stdout.
    seed_pose:
        Optional dict with keys ``base``, ``p1``, ``p2``, ``p3``, ``roll``,
        ``grip`` to pre-seed the integrated position (avoids a startup jump
        if the arm is not at the neutral pose when engagement begins).
    config:
        Optional AppConfig reference to sync limits, rates and exponents.
    """
    if not _PYGAME_AVAILABLE:
        if verbose:
            print("[ps3_joystick] pygame not installed – falling back to gesture mode.")
        return None

    try:
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
    except Exception as exc:
        if verbose:
            print(f"[ps3_joystick] pygame init failed ({exc}) – gesture mode.")
        return None

    count = pygame.joystick.get_count()
    if count == 0:
        if verbose:
            print("[ps3_joystick] No joystick detected – gesture mode active.")
        return None

    if index >= count:
        if verbose:
            print(f"[ps3_joystick] Joystick index {index} not found "
                  f"({count} available) – gesture mode.")
        return None

    try:
        js = pygame.joystick.Joystick(index)
        js.init()
    except Exception as exc:
        if verbose:
            print(f"[ps3_joystick] Could not open joystick {index}: {exc} – gesture mode.")
        return None

    if verbose:
        print(f"[ps3_joystick] ✓ Joystick: '{js.get_name()}' "
              f"axes={js.get_numaxes()} buttons={js.get_numbuttons()} "
              f"hats={js.get_numhats()}")
        print("[ps3_joystick] Control layout:")
        print("  Left Stick L/R  → Base yaw")
        print("  Left Stick U/D  → Lower pitch (p1)")
        print("  Right Stick U/D → Upper pitch (p3)")
        print("  L1 / R1         → Wrist / Roll (Left/Right)")
        print("  L2 / R2         → Gripper (Close/Open)")
        print("  D-Pad L/R       → Wrist/Roll fine trim")
        print("  SELECT          → E-Stop toggle")
        print("  START           → Freeze toggle")
        print("  PS/Guide        → Go home")
        print("  L3              → Reset joints to home pose")

    seed = seed_pose or {}
    driver = JoystickDriver(
        js,
        config = config,
        seed_base = seed.get("base",  90.0),
        seed_p1   = seed.get("p1",    45.0),
        seed_p2   = seed.get("p2",   112.5),
        seed_p3   = seed.get("p3",    90.0),
        seed_roll = seed.get("roll",  90.0),
        seed_grip = seed.get("grip",   1.0),
    )
    return driver


def shutdown_joystick() -> None:
    """Clean up pygame. Call this in the finally block of the main loop."""
    if _PYGAME_AVAILABLE and pygame.get_init():
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
