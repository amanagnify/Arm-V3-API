# Robot Arm V3 Firmware + Gesture Teleop

This project contains the Zephyr firmware for the ESP32 robot arm controller and
the Python gesture teleop host in `gesture_teleop/`.

The active wire protocol is V3:

- Port `4210`
- Raw TCP 19-byte binary frames, or WebSocket binary frames
- Magic/version: `0xA6 0x03`
- Modes: `A` active, `H` hold, `M` home
- Six absolute servo angles: base, p1, p2, p3, roll, gripper

## Firmware

From a Zephyr shell with this folder as the app:

```bash
west build -b esp32_devkitc_wroom/esp32/procpu . -p always
west flash
west monitor
```

Use the exact board name that matches your Zephyr ESP32 setup if it differs.
On boot, the firmware initializes the PCA9685, parks at the V3 home pose, joins
WiFi, prints the ESP32 IP address, and listens on TCP/WebSocket port `4210`.

The packet watchdog is **1000 ms** (was 250 ms): it is a link-loss stop, not a
stream-pacing check. Host-side hiccups of a few hundred ms are normal and must
not put the arm into hold-and-resume stutter (that showed up as "watchdog: no
packet" spam on the serial monitor).

Expected monitor lines include:

```text
PCA9685 initialized (300.0 Hz)
servo_control: V3 ready, parked at home
wifi_manager: WiFi connected
ESP32 IP: <device-ip>
tcp_transport: listening on port 4210
```

## Python Host

Create an environment and install the host dependencies:

```bash
cd gesture_teleop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_pipeline.py
```

Set the ESP32 address in `config/calibration.json` under `websocket.host` or pass
it on the command line:

```bash
python main.py --transport websocket --ws-host <device-ip> --ws-port 4210 --camera 0
```

Raw TCP is also supported:

```bash
python main.py --transport tcp --tcp-host <device-ip> --tcp-port 4210 --camera 0
```

For vision/UI only, add `--no-transport`.

### Latency & camera

The host grabs frames on a background thread and always processes the newest one,
which removes the Linux V4L2 "stale frame" lag. On startup it prints the actual
negotiated camera mode, e.g.:

```text
[camera] index=0 backend=v4l2 mode=1280x720@30 fourcc=MJPG mjpg=on threaded=on
```

Tuning knobs (CLI or `config/calibration.json` → `camera` / `vision`):

- `--profile` — print per-stage frame timings (capture / mediapipe / control / overlay).
- `--no-threaded-capture` — fall back to synchronous reads.
- `camera.use_mjpg` (default `true`) — request MJPG; needed for 720p@30 on most webcams.
- `camera.fill_window` (default `false`) — crop the feed to fill the window instead of letterboxing.
- `vision.model_complexity` (`0` = low-latency, `1` = accurate) — the biggest CPU-latency lever.

Packets are streamed by a dedicated **sender thread** (`TransportPump`) at
`write_hz`, decoupled from the camera loop: a MediaPipe stall can no longer
starve the firmware watchdog (the pump re-sends the last pose as a keepalive),
and no socket operation ever runs on the vision loop. The firmware also evicts
clients that go silent for 5 s, so a crashed/forgotten host instance can't hold
the single client slot forever — note that **only one client can be connected
at a time**: a still-running `main.py` (even a forgotten terminal) will grab
the connection and lock out a new one.

A disconnected or still-booting ESP32 no longer freezes the window: connect
attempts (including the WebSocket handshake) run on a **background thread**,
throttled to one per `reconnect_interval_s` — the old in-loop connect caused a
~0.6 s feed stall every retry until the board came up. After connecting, sends
use a tight 80 ms socket timeout, so a WiFi burp drops one packet (the firmware
profile bridges it) instead of stalling the feed. MediaPipe is also warmed up
on a dummy frame at startup so the first live frames don't pay for graph
compilation, and the control `dt` is bounded at 100 ms so a stalled frame can
never integrate into a yaw/pitch lurch.

### Two-hand control scheme (default)

The screen is split at the midline (mirrored view, so it matches your hands):

| | LEFT-half hand | RIGHT-half hand |
|---|---|---|
| base yaw | ✔ (x-deflection) | ✔ (x-deflection, contributions add) |
| p1 (reach) | | ✔ hand distance (closer = fold in) |
| p2 (height) | | ✔ hand height (direct — no more average) |
| p3 (height) | ✔ hand height | |
| roll | ✔ hand tilt | |
| gripper | ✔ curl (fist = open, palm = closed) | |

- **Neutral points move ONLY when you press `R` or `L`** (or `n` for both
  visible hands). Pressing it re-centers that hand's anchor to your current hand
  **and glides the arm to a known neutral pose**: right neutral → `p1 = 45°`,
  `p2 = 90°`; left neutral → `p3 = 90°`, `roll = 90°`, and (with an open left
  palm) the gripper opens. Anchors never drift while you work.
- **Right fist (hold 0.2 s) = PAUSE** p1/p2 (and the right hand's yaw). p1/p2 are
  held exactly where they were — they do **not** drift. The pause **survives the
  right hand leaving the frame**: bring it back and it is still paused. To
  **resume you must OPEN → CLOSE → OPEN** the right fist — a deliberate double
  gesture, so relaxing your open hand (or re-appearing) never resumes by accident.
  While paused the left hand keeps full control.
- **Gripper is natural**: left hand open = gripper open, left fist = gripper
  closed. The left hand's grip no longer affects p3.
- The input bars show **R HEIGHT** (right hand → p2) and **L HEIGHT** (left hand
  → p3) separately, since height drives a different joint for each hand.
- **Anti-jitter**: each hand's position/size/tilt runs through a One-Euro filter
  before it becomes a joint target (heavy smoothing when the hand is still, low
  lag when it moves), and a short tracking-dropout grace
  (`two_hand.track_grace_s`, 0.3 s) coasts through one-frame MediaPipe flickers
  so motion never stalls and catch-up jumps.
- **Right-hand peace / V sign (hold 0.6 s) = go home** (right hand only, so it
  can't be confused with a closed grip).
- The UI shows role-colored skeletons (R=green, L=amber), two Agnisys-logo
  anchor markers labeled L/R, a deflection line from each anchor to its hand,
  the center divider, and an R-LOCK chip while paused.
- Set `control.scheme: "direct"` in `config/calibration.json` for the legacy
  one-hand mapping.

### Motion smoothness (p1/p2/p3)

Both sides have selectable motion profiles so strategies can be A/B-tested one
at a time:

- **Host** (`config` → `motion.pitch_profile`): `s_curve` (default,
  target-velocity feedforward — fixes the accel-slow-accel pattern),
  `exp_smooth`, `trapezoid` (legacy baseline), `none` (host sends raw targets;
  firmware does all smoothing). Per-joint limits under `motion.p1/p2/p3`
  (rate/accel/jerk; p2 ships with the lowest acceleration since it couples
  with p1 and p3).
- **Firmware** (`src/servo_control.c` → `PROFILE_MODE`): `PROFILE_S_CURVE`
  (default), `PROFILE_EXP_TRACK`, `PROFILE_TRAPEZOID` (baseline). Prints a
  2 Hz `MOTION:` diagnostic line while joints are moving.
- **Pose transitions** (home↔active) use a minimum-jerk trajectory with
  coordinated arrival (`control.transition_profile`, `min_jerk`/`linear`).

Repeatable large-move test (no camera needed) for judging smoothness on the
real arm — watch the serial monitor while it runs:

```bash
python motion_test.py --host <device-ip> --mode step    # firmware-only smoothing
python motion_test.py --host <device-ip> --mode ramp --rate 10   # host min-jerk
python main.py ... --motion-log motion.csv              # target-vs-actual CSV
```

## Quick Diagnostics

Open the board monitor and confirm `STATE:` lines show absolute servo targets:

```bash
cd gesture_teleop
python read_servo_angles.py --list-ports
python read_servo_angles.py --port COM5
```

You can also open `http://<device-ip>:4210` in a browser. The firmware serves a
small WebSocket test page with V3 sliders.

## Power Notes

Use a separate 5-6 V servo rail sized for stall current, common ground with the
ESP32, and a bulk capacitor near PCA9685 V+. Brownouts look exactly like jitter.
