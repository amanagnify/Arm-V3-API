from __future__ import annotations

import math
from dataclasses import dataclass


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def min_jerk_s(tau: float) -> float:
    """Minimum-jerk position shape s(tau) = 10t^3 - 15t^4 + 6t^5 for tau in
    [0,1]: zero velocity AND zero acceleration at both ends, peak velocity
    1.875x the average. The standard profile for smooth point-to-point moves."""
    t = clamp(tau, 0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def apply_signed_deadband(value: float, deadband: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0
    scaled = (magnitude - deadband) / max(1e-6, 1.0 - deadband)
    return scaled if value >= 0.0 else -scaled


@dataclass
class LowPassFilter:
    alpha: float
    value: float | None = None

    def update(self, measurement: float, dt: float | None = None) -> float:
        # dt is accepted (and ignored) so LowPassFilter and OneEuroFilter share
        # a uniform update(x, dt) signature and are interchangeable in main.
        if self.value is None:
            self.value = measurement
        else:
            self.value += self.alpha * (measurement - self.value)
        return self.value

    def reset(self, measurement: float | None = None) -> None:
        self.value = measurement


@dataclass
class OneEuroFilter:
    """1-Euro adaptive low-pass filter (Casiez et al.). Smooths hard when the
    signal is still and lets it through with low lag when it moves fast."""

    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0
    value: float | None = None
    _x_prev: float | None = None
    _dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, measurement: float, dt: float | None = None) -> float:
        step = dt if (dt is not None and dt > 0.0) else 1e-2
        if self._x_prev is None:
            self._x_prev = measurement
            self._dx_prev = 0.0
            self.value = measurement
            return measurement
        dx = (measurement - self._x_prev) / step
        a_d = self._alpha(self.d_cutoff, step)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, step)
        x_hat = a * measurement + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self.value = x_hat
        return x_hat

    def reset(self, value: float | None = None) -> None:
        self._x_prev = value
        self._dx_prev = 0.0
        self.value = value


def make_input_filter(cfg, alpha: float = 0.4, min_cutoff: float | None = None,
                      beta: float | None = None) -> "LowPassFilter | OneEuroFilter":
    """Build the configured input filter for one signal. Both variants expose
    update(x, dt), .value and reset(), so they are interchangeable in main.
    `alpha` is used only in lowpass mode (per-axis smoothing strength).
    `min_cutoff`/`beta` override the shared One-Euro params for one signal
    (used to smooth the noisy depth channel harder than yaw/height)."""
    fcfg = cfg.filters
    if getattr(fcfg, "filter_mode", "lowpass") == "one_euro":
        return OneEuroFilter(
            min_cutoff=fcfg.one_euro_min_cutoff if min_cutoff is None else min_cutoff,
            beta=fcfg.one_euro_beta if beta is None else beta,
            d_cutoff=fcfg.one_euro_dcutoff,
        )
    return LowPassFilter(alpha)


@dataclass
class SlewRateLimiter:
    rate_per_second: float
    value: float | None = None

    def update(self, target: float, dt: float, rate_scale: float = 1.0) -> float:
        if self.value is None:
            self.value = target
            return target
        max_step = self.rate_per_second * max(0.0, rate_scale) * max(1e-3, dt)
        delta = clamp(target - self.value, -max_step, max_step)
        self.value += delta
        return self.value

    def reset(self, value: float | None = None) -> None:
        self.value = value


# S-curve tuning shared by all joints (per-joint vel/acc/jerk come from config)
_APPROACH_GAIN = 4.0  # 1/s — proportional landing zone gain near zero error


@dataclass
class SCurveLimiter:
    """S-curve tracker with TARGET-VELOCITY FEEDFORWARD.

    Fixes the accelerate-slow-accelerate limit cycle of a plain trapezoid
    chasing a streamed (moving) target: a stop-distance brake law re-plans a
    full stop every tick, so the follower alternately catches up and brakes.

    Structure (all-LINEAR in steady tracking, so it cannot limit-cycle):
      1. desired velocity = target_velocity(ff) + approach(error)
         (approach: sqrt curve far out, proportional near zero)
      2. first-order lag on the velocity command — this is what shapes the
         acceleration into an S; its time constant is max_accel/max_jerk
      3. hard acceleration cap as a safety clip (inactive in steady tracking)
    """

    max_rate: float          # deg/s
    max_accel: float         # deg/s^2
    max_jerk: float          # deg/s^3 (sets the accel-shaping lag: acc/jerk)
    value: float | None = None
    velocity: float = 0.0
    _v_cmd: float = 0.0
    _prev_target: float | None = None
    _target_vel: float = 0.0

    def _lag_tc(self) -> float:
        return clamp(self.max_accel / max(self.max_jerk, 1e-6), 0.04, 0.12)

    def update(self, target: float, dt: float, rate_scale: float = 1.0) -> float:
        if self.value is None:
            self.reset(target)
            return target
        dt = clamp(dt, 1e-3, 0.1)

        # low-passed target velocity estimate (the feedforward term)
        if self._prev_target is not None:
            raw_tv = (target - self._prev_target) / dt
            self._target_vel += 0.35 * (raw_tv - self._target_vel)
        self._prev_target = target

        max_v = self.max_rate * max(0.0, rate_scale)
        tv = clamp(self._target_vel, -max_v, max_v)
        error = target - self.value

        # snap only when everything (incl. the target itself) is quiet
        if abs(error) < 0.25 and abs(self.velocity) < 1.0 and abs(tv) < 1.0:
            self.value = target
            self.velocity = 0.0
            self._v_cmd = 0.0
            return self.value

        abs_e = abs(error)
        approach = min(math.sqrt(2.0 * (0.5 * self.max_accel) * abs_e),
                       _APPROACH_GAIN * abs_e)
        v_des = clamp(tv + (approach if error > 0.0 else -approach), -max_v, max_v)

        # linear accel shaping (S-curve), then the hard accel safety clip
        self._v_cmd += (v_des - self._v_cmd) * min(1.0, dt / self._lag_tc())
        dv = clamp(self._v_cmd - self.velocity, -self.max_accel * dt, self.max_accel * dt)
        self.velocity = clamp(self.velocity + dv, -max_v, max_v)
        self.value += self.velocity * dt
        return self.value

    def reset(self, value: float | None = None) -> None:
        self.value = value
        self.velocity = 0.0
        self._v_cmd = 0.0
        self._prev_target = value
        self._target_vel = 0.0


@dataclass
class ExpSmoothLimiter:
    """First-order tracker: velocity = gain*error + target-velocity feedforward,
    acceleration-limited. Never overshoots (first order), tracks ramps with a
    constant small lag, decelerates exponentially into the target."""

    max_rate: float          # deg/s
    gain: float = 5.0        # 1/s — proportional velocity gain
    max_accel: float = 400.0  # deg/s^2
    value: float | None = None
    velocity: float = 0.0
    _prev_target: float | None = None
    _target_vel: float = 0.0

    def update(self, target: float, dt: float, rate_scale: float = 1.0) -> float:
        if self.value is None:
            self.reset(target)
            return target
        dt = clamp(dt, 1e-3, 0.1)

        if self._prev_target is not None:
            raw_tv = (target - self._prev_target) / dt
            self._target_vel += 0.35 * (raw_tv - self._target_vel)
        self._prev_target = target

        max_v = self.max_rate * max(0.0, rate_scale)
        tv = clamp(self._target_vel, -max_v, max_v)
        error = target - self.value

        if abs(error) < 0.15 and abs(self.velocity) < 1.0 and abs(tv) < 1.0:
            self.value = target
            self.velocity = 0.0
            return self.value

        v_des = clamp(tv + self.gain * error, -max_v, max_v)
        dv = clamp(v_des - self.velocity, -self.max_accel * dt, self.max_accel * dt)
        self.velocity += dv
        self.value += self.velocity * dt
        return self.value

    def reset(self, value: float | None = None) -> None:
        self.value = value
        self.velocity = 0.0
        self._prev_target = value
        self._target_vel = 0.0


@dataclass
class AccelSlewLimiter:
    """Velocity- AND acceleration-limited tracker (host-side trapezoidal profile,
    mirroring the firmware's smoother). Unlike SlewRateLimiter it always starts
    moving slowly from standstill, which is what makes pitch motion feel gentle.
    `rate_scale` scales the velocity cap (dynamic slew / live-speed multiplier).
    NOTE: kept as the 'trapezoid' baseline; when chasing a continuously moving
    target it can limit-cycle (accel/brake) — prefer SCurveLimiter for that."""

    max_rate: float           # deg/s
    max_accel: float          # deg/s^2
    value: float | None = None
    velocity: float = 0.0

    def update(self, target: float, dt: float, rate_scale: float = 1.0) -> float:
        if self.value is None:
            self.value = target
            self.velocity = 0.0
            return target
        dt = max(1e-3, dt)
        max_v = self.max_rate * max(0.0, rate_scale)
        accel_mag = max(self.max_accel, 1e-6)
        error = target - self.value

        # Snap when essentially there and slow, to avoid limit-cycle dithering.
        if abs(error) < 0.5 and abs(self.velocity) < 1.0:
            self.value = target
            self.velocity = 0.0
            return self.value

        stop_distance = (self.velocity * self.velocity) / (2.0 * accel_mag)
        if abs(error) <= stop_distance:
            accel = -accel_mag if self.velocity > 0.0 else accel_mag
        else:
            accel = accel_mag if error > 0.0 else -accel_mag

        self.velocity = clamp(self.velocity + accel * dt, -max_v, max_v)
        self.value += self.velocity * dt
        return self.value

    def reset(self, value: float | None = None) -> None:
        self.value = value
        self.velocity = 0.0
