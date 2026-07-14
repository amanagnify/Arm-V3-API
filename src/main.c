/* Robot Servo TCP firmware */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "app_callbacks.h"
#include "servo_control.h"
#include "tcp_transport.h"
#include "websocket_layer.h"
#include "wifi_manager.h"

#define TCP_LISTEN_PORT 4220

#define CONTROL_PERIOD_MS 10
#define CONTROL_THREAD_STACK_SIZE 2048
#define CONTROL_THREAD_PRIORITY 5

static K_SEM_DEFINE(control_tick_sem, 0, 1);
static struct k_timer control_timer;
static struct k_mutex control_lock;

K_THREAD_STACK_DEFINE(control_thread_stack, CONTROL_THREAD_STACK_SIZE);
static struct k_thread control_thread_data;

static void control_timer_handler(struct k_timer *timer_id)
{
	ARG_UNUSED(timer_id);
	k_sem_give(&control_tick_sem);
}

static void control_thread_fn(void *a, void *b, void *c)
{
	ARG_UNUSED(a);
	ARG_UNUSED(b);
	ARG_UNUSED(c);

	while (1) {
		k_sem_take(&control_tick_sem, K_FOREVER);

		k_mutex_lock(&control_lock, K_FOREVER);
		servo_control_update_outputs();
		k_mutex_unlock(&control_lock);
	}
}

int main(void)
{
	printk("Robot Servo firmware start\n");

	if (servo_control_init() != 0) {
		return -1;
	}

	if (wifi_manager_init() != 0) {
		return -1;
	}
	while (wifi_manager_connect() != 0) {
		printk("WiFi connect failed, retrying in 2 seconds...\n");
		k_sleep(K_SECONDS(2));
	}

	k_mutex_init(&control_lock);
	k_timer_init(&control_timer, control_timer_handler, NULL);

	/* Servo control thread, ticked every CONTROL_PERIOD_MS (10 ms = 100 Hz) */
	k_thread_create(&control_thread_data, control_thread_stack,
			K_THREAD_STACK_SIZEOF(control_thread_stack), control_thread_fn, NULL,
			NULL, NULL, CONTROL_THREAD_PRIORITY, 0, K_NO_WAIT);

	k_timer_start(&control_timer, K_NO_WAIT, K_MSEC(CONTROL_PERIOD_MS));

	app_ctx_t app_ctx = {
		.control_lock = &control_lock,
		.protocol_detected = false,
		.use_websocket = false,
		.ws_raw_len = 0,
		.ws_state = WS_STATE_CLOSED,
	};

	if (tcp_transport_start(TCP_LISTEN_PORT, app_on_tcp_connect, app_on_tcp_data,
				app_on_tcp_disconnect, &app_ctx) != 0) {
		return -1;
	}

	/* Heartbeat loop */
	while (1) {
		k_sleep(K_SECONDS(1));

		if (!wifi_manager_is_connected()) {
			printk("WiFi lost, reconnecting...\n");
			if (wifi_manager_reconnect_if_needed() == 0) {
				(void)tcp_transport_restart();
			}
		}
	}
}
