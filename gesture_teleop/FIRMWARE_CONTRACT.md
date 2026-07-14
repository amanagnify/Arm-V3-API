# Firmware Contract (host → Zephyr firmware) — V3, revision 2

The Python host (`gesture_teleop/`) speaks **protocol V3**. The firmware
(`../src/`, Zephyr) now **implements V3** — this document describes the agreed
behavior and is kept in sync with `../src/servo_control.c` and
`../src/app_callbacks.c`. The Python reference implementation is
[`protocol.py`](protocol.py); `test_pipeline.py` (`test_firmware_parity`)
runs a faithful port of the firmware intake and cross-checks it against host
packets, so host↔firmware compatibility is verified automatically.

**Changes in revision 2** (vs the first V3 hand-off):
1. The **gripper channel now carries an absolute servo angle** (open = 166.5°,
   closed = 36.0°), not `fraction*180`. All six channels decode identically.
2. **p2 direction is configurable on the host**. In the current calibration
   `p2_invert` is `false`, so p2 is sent in direct joint space (range 45–180).
   Firmware must NOT invert it again.
3. **Home pose is now specified exactly** (energy-efficient park position).
4. `H` (hold) semantics clarified; mode `'S'` is gone — the host never sends it.
5. New per-joint clamps (p1 up to 100° so the home pose is reachable).
6. A dedicated **anti-jitter / smooth-motion checklist** (see §7).

---

## 1. Transport
- Unchanged: **TCP** (raw binary) or **WebSocket** (binary frames, opcode 0x02)
  on port **4210**. The host auto-frames WS; raw TCP receives the bytes directly.
- The `serial` and `udp` host transports are **parked** (no firmware counterpart).
- Host streams at ~30 Hz (WebSocket) / up to 60 Hz (TCP).
- On the TCP raw stream, resynchronize by scanning for magic `0xA6` and
  validating a full **19-byte** frame (note: length changed from 18 in V2).

## 2. V3 control packet — 19 bytes, big-endian
```
byte 0     MAGIC   = 0xA6
byte 1     VERSION = 0x03
byte 2..3  sequence (uint16)
byte 4     mode    : 'A' active | 'H' hold | 'M' home
byte 5     mask    = 0x3F  (all six joints present)
byte 6..7   base    q16
byte 8..9   p1      q16   (lower pitch)
byte 10..11 p2      q16   (middle pitch, SERVO space — already flipped by host)
byte 12..13 p3      q16   (upper pitch)
byte 14..15 roll    q16
byte 16..17 gripper q16   (ABSOLUTE servo angle: open=166.5, closed=36.0)
byte 18     checksum = XOR of bytes 0..17
```
**Every channel is an absolute servo angle in [0,180]. Uniform decode:**
```
angle = q / 65535 * 180
```
Reject a packet if: length != 19, magic/version/mask wrong, mode not in
{A,H,M}, or the XOR checksum fails. On rejection, keep the previous target
(do not twitch), optionally log.

## 3. Per-joint clamps (firmware owns final safety clamping)
Clamp each decoded angle to the joint's mechanical limit before driving:

| channel | limit (deg) | active pose | home pose | notes |
|---------|-------------|-------------|-----------|-------|
| base yaw | 0 – 180 | 90 | 90 | absolute position — do NOT integrate velocity |
| p1 (lower pitch) | 0 – 100 | 45 | 98.1 | teleop stays ≤ 90; 100 allows the home pose |
| p2 (middle pitch) | 45 – 180 | 135 | 88.2 | host-side `p2_invert` is disabled in this calibration; firmware must not flip it again |
| p3 (upper pitch) | 0 – 180 | 45 | 90 | host teleop range reduced to 0–90 for control |
| roll | 0 – 180 | 90 | 90 | |
| gripper | 36 – 166.5 | 166.5 (open) | 166.5 (open) | absolute angle; 166.5=open, 36=closed |

Servos are direct-drive (gear ratio 1); apply any per-servo zero/direction
trim here — and keep the p2 host calibration in sync with the motor wiring.

## 4. Mode semantics
- **`A` active** — track the packet's six angles (through your motion profile).
- **`H` hold** — **hold the last commanded pose exactly** (do NOT go home).
  Used by the host for freeze, e-stop, and the 10 s tracking-loss hold. This is
  also the required **watchdog** behavior.
- **`M` home** — move to the firmware's home pose using a SLOW profile
  (~10 °/s per joint). The host sends `M` while parked/waiting at boot; it
  normally drives home↔active transitions itself with mode-`A` ramps at
  ~5 °/s, so under `A` the firmware must simply track smoothly.

## 5. Home pose & startup
```
base 90.0   p1 98.1   p2 88.2   p3 90.0   roll 90.0   gripper 166.5 (open)
```
- On boot: engage servos at their resting pulses (no snap), then move SLOWLY
  (~10 °/s) to the home pose above, then hold. The host will later ramp the
  arm to the active pose (base 90 / p1 45 / p2 135 / p3 45 / roll 90 /
  gripper 166.5) via mode-`A` packets at ~5 °/s once a hand is detected.
- Keep the **1000 ms watchdog**: if no valid packet for 1000 ms → behave as `H`
  (hold last pose). Never jump or go home on comms loss. Do NOT make it
  tighter: the host camera loop legitimately hiccups for a few hundred ms
  (MediaPipe scheduling), and a tight watchdog turns every hiccup into a
  visible hold-and-resume stutter on the arm.

## 6. What the host guarantees
- All six values are already smoothed (One-Euro input filters + host-side
  velocity/acceleration-limited slew) and clamped to the table in §3.
- Base yaw and roll are integrated to absolute angles host-side with deadband
  and gradual acceleration — consecutive packets differ by small deltas.
- During pose transitions the host moves every joint ≤ ~5 °/s (operator
  tunable 1–30 °/s).
- Sequence increments per packet (wraps at 65535); duplicates may occur — the
  firmware may ignore repeated identical payloads.

## 7. Smooth-motion / anti-jitter checklist (implemented in servo_control.c)
1. **Motion profile — IMPORTANT, replaces the plain trapezoid.** A trapezoid
   with stop-distance braking chasing a STREAMED (moving) target limit-cycles:
   it catches up, enters its braking distance, brakes, the target pulls ahead,
   it re-accelerates — that is the "accelerate-slow-accelerate" seen on large
   p1/p2/p3 moves. The firmware now has three compile-selectable profiles
   (`PROFILE_MODE` in `servo_control.c`; test one at a time):
   - `PROFILE_TRAPEZOID` — legacy baseline, kept for A/B comparison.
   - `PROFILE_S_CURVE` (default) — **target-velocity feedforward** + sqrt/linear
     approach + a first-order lag on the velocity command (accel becomes
     S-shaped; lag = accel/jerk). All-linear while tracking, so it cannot
     limit-cycle; the hard accel cap is only a safety clip. Verified by the
     host-side regression `test_pipeline.py::test_s_curve`.
   - `PROFILE_EXP_TRACK` — first-order tracker (vel = 5·error + feedforward);
     never overshoots; simplest fallback.
   **Per-joint limits** (`joint_profiles[]`, vel °/s / acc °/s² / jerk °/s³):
   base 60/150/900 · p1 45/90/450 (loaded) · **p2 40/60/300 (lowest accel —
   couples with both p1 and p3)** · p3 50/110/600 · roll 80/200/1200 ·
   gripper 100/400/2500. Mode `M`/boot velocity cap: 10 °/s.
   The feedforward estimate comes from per-packet target deltas (low-passed,
   α=0.35) and decays when the stream goes stale (>150 ms) so the profile
   settles instead of coasting.
2. **Deadband target updates**: ignore a new target that differs from the
   current one by less than ~0.2°. This kills flicker from 16-bit quantization
   and hand micro-tremor (1 LSB ≈ 0.003°, but filtered inputs can dither a
   few tenths of a degree).
3. **Round, don't truncate, the pulse math.** In `servo_control.c` the pulse is
   computed as `min_us + (uint16_t)(norm * (max_us - min_us))` — the cast
   TRUNCATES and adds up to 1 µs of asymmetric error (~0.09°) that toggles
   between frames. Compute in float and round:
   `pulse = min_us + (uint16_t)(norm * (max_us - min_us) + 0.5f)`.
4. **Keep the same-value I²C write suppression** in `servo.c` (`last_pwm`
   check) — with (2) and (3) it eliminates useless bus traffic entirely when
   the arm is still.
5. **No logging inside the 100 Hz control path.** `printk` in
   `servo_control_update_outputs()` (the GESTURE line) can stall the loop and
   cause visible stutter — rate-limit it hard (≥ 250 ms, already partly done)
   or move it to a lower-priority thread.
6. **dt hygiene**: keep the `dt` clamp in the control loop, and make sure the
   control timer period matches the comment (it is 10 ms, not 2 ms).
7. **Power**: most "jitter" on this class of arm is electrical. Servo rail
   must be a separate 5–6 V supply able to deliver stall current, with common
   ground to the ESP32 and a bulk capacitor (≥ 470 µF) at the PCA9685 V+.
   Brownouts during simultaneous multi-joint moves look exactly like firmware
   jitter.
8. **PWM resolution**: at 300 Hz the 12-bit PCA9685 gives ~0.8 µs/tick
   (~0.07°) — adequate; no change needed. If you ever drop to 50 Hz, expect
   ~4.9 µs/tick (~0.44°) steps, which WILL look jittery with slow motion.
9. **Do not re-integrate base velocity.** V3 base is an absolute angle; the
   old V2 "joystick" integration must be removed.

## 8. Suggested firmware-side constants (mirror of §3)
```c
#define PKT_MAGIC        0xA6
#define PKT_VERSION_V3   0x03
#define PKT_LEN_V3       19
#define PKT_MASK_ALL     0x3F

/* clamps */
base:    0.0 .. 180.0    p1: 0.0 .. 100.0    p2: 45.0 .. 180.0
p3:      0.0 .. 180.0    roll: 0.0 .. 180.0  gripper: 36.0 .. 166.5

/* home pose */
{90.0f, 98.1f, 88.2f, 90.0f, 90.0f, 166.5f}

/* watchdog */ 1000 ms -> hold last pose (mode 'H' behavior)
```

## 9. Reference
`protocol.build_packet()` / `protocol.parse_packet()` in
[`protocol.py`](protocol.py) are authoritative for the byte layout, and
`test_pipeline.py` (`python test_pipeline.py`) round-trips every channel,
the checksum, and the clamping behavior.
