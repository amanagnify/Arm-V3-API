#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct k_mutex;

typedef struct {
	struct k_mutex *control_lock;
	bool use_websocket;
	bool protocol_detected;

	/* TCP raw stream reassembly: 64 bytes holds 3 complete 19-byte V3 packets */
	uint8_t tcp_raw_buf[64];
	size_t tcp_raw_len;

	/* WebSocket frame reassembly */
	uint8_t ws_raw_buf[1024];
	size_t ws_raw_len;

	/* WebSocket handshake / HTTP response headers */
	uint8_t ws_tx_buf[256];

	/* Decoded WS payload (one 19-byte packet + margin) */
	uint8_t ws_payload_buf[256];
	int ws_state; /* ws_state_t */
} app_ctx_t;

void app_on_tcp_data(const uint8_t *data, size_t len, void *ctx);
void app_on_tcp_connect(void *ctx);
void app_on_tcp_disconnect(void *ctx);
