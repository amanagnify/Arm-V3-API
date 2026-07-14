/* Servo control for the 6-DOF arm — protocol V3 (FIRMWARE_CONTRACT.md rev 2).
 *
 * Every packet channel is an ABSOLUTE servo angle in [0,180] degrees,
 * including the gripper (open=166.5, closed=36.0) and p2 (pre-flipped by the
 * host — do NOT invert it here). The base is an absolute position; the old
 * V2 velocity integration is gone.
 *
 * Modes:  'A' track the packet angles   'H' hold the last commanded pose
 *         'M' move to the home pose with the slow profile
 * Watchdog (1000 ms without a valid packet) behaves exactly like 'H'. It is a
 * link-loss stop, NOT a stream-pacing check: the host camera loop can hiccup
 * for a few hundred ms (MediaPipe, OS scheduling) and that must not trip it —
 * the profile already settles smoothly on the last target by itself.
 *
 * MOTION PROFILES (compile-time selectable, test one at a time):
 *   PROFILE_TRAPEZOID — legacy baseline. Stop-distance braking. When chasing
 *       a STREAMED (moving) target it limit-cycles: catch up -> enter braking
 *       distance -> brake -> target pulls ahead -> re-accelerate. That is the
 *       "accelerate, slow, accelerate" seen on large p1/p2/p3 moves.
 *   PROFILE_S_CURVE  — default. Jerk-limited with TARGET-VELOCITY FEEDFORWARD:
 *       commanded velocity = target_velocity + sqrt-approach(error). Tracks a
 *       moving target at the target's own speed (no re-planned stops), lands
 *       on a stationary target critically damped. Smoothest under load.
 *   PROFILE_EXP_TRACK — first-order tracker (vel = gain*error + feedforward),
 *       acceleration-limited. Never overshoots; simplest fallback.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "servo.h"
#include "servo_control.h"

#define NUM_SERVOS 6 /* packet/channel order: base, p1, p2, p3, roll, gripper */

#define WATCHDOG_TIMEOUT_MS 1000

/* ================ MOTION PROFILE SELECTION ================ */
#define PROFILE_TRAPEZOID 0
#define PROFILE_S_CURVE 1
#define PROFILE_EXP_TRACK 2

#ifndef PROFILE_MODE
#define PROFILE_MODE PROFILE_S_CURVE
#endif

#define EXP_TRACK_GAIN 5.0f /* 1/s, PROFILE_EXP_TRACK only */

/* Low-rate motion diagnostics (2 Hz, only while moving). 0 to silence. */
#ifndef MOTION_DIAG
#define MOTION_DIAG 1
#endif

/* ================ V3 PACKET ================ */
#define PKT_MAGIC 0xA6U
#define PKT_VERSION 0x03U
#define PKT_LEN 19
#define PKT_MASK_ALL 0x3FU

/* ================ PER-JOINT LIMITS (deg) ================ */
static const float joint_min_deg[NUM_SERVOS] = {0.0f, 0.0f, 45.0f, 0.0f, 0.0f, 36.0f};
static const float joint_max_deg[NUM_SERVOS] = {180.0f, 100.0f, 180.0f, 180.0f, 180.0f, 166.5f};

/* Energy-efficient park position. The arm is expected to be physically parked
 * here at power-up, so commanding it at boot produces no snap. */
static const float home_pose_deg[NUM_SERVOS] = {90.0f, 98.1f, 88.2f, 90.0f, 90.0f, 166.5f};

/* ================ PER-JOINT MOTION TUNING ================
 * Loaded joints get gentler limits: p1 carries the arm (60 kgcm servo but the
 * full link inertia), p2 couples with BOTH p1 and p3 so it gets the lowest
 * acceleration, p3 is medium. base/roll/grip are lightly loaded. */
struct joint_profile {
	float max_vel;  /* deg/s   */
	float max_acc;  /* deg/s^2 */
	float max_jerk; /* deg/s^3 (s-curve only) */
};

static const struct joint_profile joint_profiles[NUM_SERVOS] = {
	{60.0f, 150.0f, 900.0f},   /* base */
	{45.0f, 90.0f, 450.0f},    /* p1 (loaded) */
	{40.0f, 60.0f, 300.0f},    /* p2 (lowest accel: couples p1+p3) */
	{50.0f, 110.0f, 600.0f},   /* p3 */
	{80.0f, 200.0f, 1200.0f},  /* roll */
	{100.0f, 400.0f, 2500.0f}, /* gripper */
};

#define HOME_SPEED_DEG_S 10.0f    /* velocity cap for mode 'M' and boot */
#define TARGET_DEADBAND_DEG 0.2f  /* ignore sub-quantization target flicker */
#define TARGET_VEL_LP_ALPHA 0.35f /* feedforward estimator smoothing */
#define APPROACH_GAIN 4.0f        /* 1/s, proportional landing zone (s-curve) */

/* Uniform pulse map: 0..180 deg -> 500..2500 us on every channel. */
#define SERVO_MIN_US 500.0f
#define SERVO_MAX_US 2500.0f

struct joint_state {
	float pos;        /* profiled output angle */
	float vel;        /* profiled velocity */
	float acc;        /* trapezoid accel state (baseline mode only) */
	float v_cmd;      /* s-curve: lagged velocity command (accel shaping) */
	float target;     /* latest commanded angle */
	float target_vel; /* low-passed target velocity (feedforward) */
};

static struct joint_state joints[NUM_SERVOS];

static uint16_t last_sequence;
static char last_mode = 'M';
static bool got_first_packet;
static bool watchdog_logged;
static int64_t last_packet_ms;
static int64_t last_update_ms;

/* ================ HELPERS ================ */
static float clampf(float value, float lo, float hi)
{
	if (value < lo) {
		return lo;
	}
	if (value > hi) {
		return hi;
	}
	return value;
}

static uint16_t read_u16_be(const uint8_t *p)
{
	return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static float dequantize_angle(uint16_t q)
{
	return ((float)q / 65535.0f) * 180.0f;
}

static uint16_t angle_to_pulse_us(float pos_deg)
{
	float us = SERVO_MIN_US + (pos_deg / 180.0f) * (SERVO_MAX_US - SERVO_MIN_US);
	return (uint16_t)(us + 0.5f); /* round — truncation causes 1-us flicker */
}

/* ================ PROFILE STEPS (one joint, one tick) ================ */

/* Legacy baseline. Kept for A/B testing; see header comment for its flaw. */
static void step_trapezoid(struct joint_state *st, const struct joint_profile *p,
			   float vcap, float dt)
{
	float error = st->target - st->pos;

	if (fabsf(error) < 0.5f && fabsf(st->vel) < 1.0f) {
		st->pos = st->target;
		st->vel = 0.0f;
		st->acc = 0.0f;
		return;
	}

	float stop_distance = (st->vel * st->vel) / (2.0f * p->max_acc);
	float accel;

	if (fabsf(error) <= stop_distance) {
		accel = (st->vel > 0.0f) ? -p->max_acc : p->max_acc;
	} else {
		accel = (error > 0.0f) ? p->max_acc : -p->max_acc;
	}

	st->vel = clampf(st->vel + accel * dt, -vcap, vcap);
	st->pos += st->vel * dt;
}

/* S-curve feedforward tracking (default).
 *
 * All-LINEAR in steady tracking, so it cannot limit-cycle (a hard jerk rate
 * limiter inside the loop is itself a limit-cycle generator — verified by the
 * host-side regression test). The velocity command is lagged by acc/jerk,
 * which shapes acceleration into an S; the hard accel cap is only a safety
 * clip that engages during big transients. */
static void step_s_curve(struct joint_state *st, const struct joint_profile *p,
			 float vcap, float dt)
{
	float error = st->target - st->pos;
	float tv = clampf(st->target_vel, -vcap, vcap);

	/* snap only when everything — including the target itself — is quiet */
	if (fabsf(error) < 0.25f && fabsf(st->vel) < 1.0f && fabsf(tv) < 1.0f) {
		st->pos = st->target;
		st->vel = 0.0f;
		st->v_cmd = 0.0f;
		return;
	}

	/* approach: sqrt far out (time-optimal-ish), proportional near zero */
	float abs_e = fabsf(error);
	float approach = fminf(sqrtf(2.0f * (0.5f * p->max_acc) * abs_e), APPROACH_GAIN * abs_e);
	float v_des = clampf(tv + ((error > 0.0f) ? approach : -approach), -vcap, vcap);

	/* linear accel shaping (the S), then the hard accel safety clip */
	float lag_tc = clampf(p->max_acc / p->max_jerk, 0.04f, 0.12f);

	st->v_cmd += (v_des - st->v_cmd) * fminf(1.0f, dt / lag_tc);
	float dv = clampf(st->v_cmd - st->vel, -p->max_acc * dt, p->max_acc * dt);

	st->vel = clampf(st->vel + dv, -vcap, vcap);
	st->pos += st->vel * dt;
}

/* First-order tracker: never overshoots, exponential landing. */
static void step_exp_track(struct joint_state *st, const struct joint_profile *p,
			   float vcap, float dt)
{
	float error = st->target - st->pos;
	float tv = clampf(st->target_vel, -vcap, vcap);

	if (fabsf(error) < 0.15f && fabsf(st->vel) < 1.0f && fabsf(tv) < 1.0f) {
		st->pos = st->target;
		st->vel = 0.0f;
		st->acc = 0.0f;
		return;
	}

	float v_des = clampf(tv + EXP_TRACK_GAIN * error, -vcap, vcap);
	float dv = clampf(v_des - st->vel, -p->max_acc * dt, p->max_acc * dt);

	st->vel += dv;
	st->pos += st->vel * dt;
}

/* ================ PACKET INTAKE ================ */
bool servo_control_apply_packet(const uint8_t *buf, int len)
{
	if (len != PKT_LEN) {
		return false;
	}
	if (buf[0] != PKT_MAGIC || buf[1] != PKT_VERSION) {
		return false;
	}

	char mode = (char)buf[4];

	if (mode != 'A' && mode != 'H' && mode != 'M') {
		return false;
	}
	if (buf[5] != PKT_MASK_ALL) {
		return false;
	}

	uint8_t checksum = 0;

	for (int i = 0; i < PKT_LEN - 1; i++) {
		checksum ^= buf[i];
	}
	if (checksum != buf[PKT_LEN - 1]) {
		return false;
	}

	int64_t now_ms = k_uptime_get();
	/* time since the previous accepted packet, for the feedforward estimate */
	float pkt_dt = (float)(now_ms - last_packet_ms) * 0.001f;

	pkt_dt = clampf(pkt_dt, 0.01f, 0.2f);

	last_sequence = read_u16_be(&buf[2]);
	last_mode = mode;
	last_packet_ms = now_ms;
	watchdog_logged = false;

	if (mode == 'A') {
		for (int i = 0; i < NUM_SERVOS; i++) {
			float angle = dequantize_angle(read_u16_be(&buf[6 + 2 * i]));

			angle = clampf(angle, joint_min_deg[i], joint_max_deg[i]);
			if (fabsf(angle - joints[i].target) >= TARGET_DEADBAND_DEG) {
				if (got_first_packet) {
					float raw_tv = (angle - joints[i].target) / pkt_dt;

					joints[i].target_vel +=
						TARGET_VEL_LP_ALPHA *
						(raw_tv - joints[i].target_vel);
				}
				joints[i].target = angle;
			}
		}
	} else if (mode == 'M') {
		for (int i = 0; i < NUM_SERVOS; i++) {
			joints[i].target = home_pose_deg[i];
			joints[i].target_vel = 0.0f;
		}
	}
	/* 'H': hold — leave the current targets untouched. */

	got_first_packet = true;
	return true;
}

/* ================ 100 Hz CONTROL LOOP ================ */
void servo_control_update_outputs(void)
{
	int64_t now_ms = k_uptime_get();
	float dt = (float)(now_ms - last_update_ms) * 0.001f;

	last_update_ms = now_ms;
	if (dt <= 0.0f || dt > 0.05f) {
		dt = 0.01f;
	}

	int64_t since_packet = now_ms - last_packet_ms;

	/* Watchdog == mode 'H': targets stay where they are; log the event once. */
	if (got_first_packet && !watchdog_logged && since_packet > WATCHDOG_TIMEOUT_MS) {
		printk("watchdog: no packet for %d ms, holding pose\n", WATCHDOG_TIMEOUT_MS);
		watchdog_logged = true;
	}

	/* No fresh packets -> the target is no longer moving: decay feedforward
	 * so the profile settles instead of coasting on a stale estimate. */
	bool stream_stale = (!got_first_packet) || since_packet > 150;

	/* Slow cap while parking (mode 'M') or before the first packet. */
	bool slow_cap = (last_mode == 'M') || !got_first_packet;

	for (int i = 0; i < NUM_SERVOS; i++) {
		struct joint_state *st = &joints[i];
		const struct joint_profile *p = &joint_profiles[i];

		if (stream_stale) {
			st->target_vel *= 0.85f;
		}

		float vcap = slow_cap ? fminf(HOME_SPEED_DEG_S, p->max_vel) : p->max_vel;

#if PROFILE_MODE == PROFILE_TRAPEZOID
		step_trapezoid(st, p, vcap, dt);
#elif PROFILE_MODE == PROFILE_EXP_TRACK
		step_exp_track(st, p, vcap, dt);
#else
		step_s_curve(st, p, vcap, dt);
#endif

		/* hard limit + anti-windup: never integrate velocity into a wall */
		float bounded = clampf(st->pos, joint_min_deg[i], joint_max_deg[i]);

		if (bounded != st->pos) {
			st->pos = bounded;
			st->vel = 0.0f;
			st->acc = 0.0f;
			st->v_cmd = 0.0f;
		}

		servo_set_pulse((uint8_t)i, angle_to_pulse_us(st->pos));
	}

#if MOTION_DIAG
	{
		static int64_t last_diag_ms;
		bool moving = false;

		for (int i = 1; i <= 3; i++) { /* p1..p3 */
			if (fabsf(joints[i].target - joints[i].pos) > 0.5f) {
				moving = true;
				break;
			}
		}
		if (moving && (now_ms - last_diag_ms) >= 500) {
			last_diag_ms = now_ms;
			printk("MOTION: p1 t=%.1f c=%.1f v=%.1f | p2 t=%.1f c=%.1f v=%.1f | "
			       "p3 t=%.1f c=%.1f v=%.1f\n",
			       (double)joints[1].target, (double)joints[1].pos,
			       (double)joints[1].vel, (double)joints[2].target,
			       (double)joints[2].pos, (double)joints[2].vel,
			       (double)joints[3].target, (double)joints[3].pos,
			       (double)joints[3].vel);
		}
	}
#endif

	servo_control_print_teleop_state();
}

int servo_control_init(void)
{
	int init_ret = -1;

	for (int attempt = 1; attempt <= 5; attempt++) {
		init_ret = servo_init();
		if (init_ret == 0) {
			break;
		}
		k_msleep(200);
	}
	if (init_ret != 0) {
		printk("servo_init failed\n");
		return -1;
	}

	/* Engage every servo at the home pose (== physical park position, so no
	 * snap), then hold. The host confirms with mode 'M' once it connects. */
	for (int i = 0; i < NUM_SERVOS; i++) {
		joints[i].pos = home_pose_deg[i];
		joints[i].vel = 0.0f;
		joints[i].acc = 0.0f;
		joints[i].v_cmd = 0.0f;
		joints[i].target = home_pose_deg[i];
		joints[i].target_vel = 0.0f;
		servo_set_pulse((uint8_t)i, angle_to_pulse_us(home_pose_deg[i]));
	}

	k_msleep(500); /* let the servos engage and settle */

	last_packet_ms = k_uptime_get();
	last_update_ms = last_packet_ms;
	printk("servo_control: V3 ready (profile=%d), parked at home\n", (int)PROFILE_MODE);
	return 0;
}

/* ================ LOGGING (rate-limited, change-gated) ================ */
void servo_control_print_teleop_state(void)
{
	static float prev_target[NUM_SERVOS] = {-1000.0f, -1000.0f, -1000.0f,
						-1000.0f, -1000.0f, -1000.0f};
	static char prev_mode = '\0';
	static int64_t last_print_ms;

	int64_t now_ms = k_uptime_get();

	if (now_ms - last_print_ms < 250) {
		return; /* never log faster than 4 Hz — keeps the control loop clean */
	}

	bool changed = (last_mode != prev_mode);

	for (int i = 0; i < NUM_SERVOS && !changed; i++) {
		if (fabsf(joints[i].target - prev_target[i]) > 0.25f) {
			changed = true;
		}
	}
	if (!changed) {
		return;
	}

	last_print_ms = now_ms;
	prev_mode = last_mode;
	for (int i = 0; i < NUM_SERVOS; i++) {
		prev_target[i] = joints[i].target;
	}

	// printk("STATE: base=%.1f p1=%.1f p2=%.1f p3=%.1f roll=%.1f grip=%.1f mode=%c seq=%u\n",
	//        (double)joints[0].target, (double)joints[1].target, (double)joints[2].target,
	//        (double)joints[3].target, (double)joints[4].target, (double)joints[5].target,
	//        last_mode, (unsigned)last_sequence);
}

uint32_t servo_control_last_sequence(void)
{
	return last_sequence;
}

char servo_control_last_mode(void)
{
	return last_mode;
}
