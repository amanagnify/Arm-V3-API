#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
	WS_STATE_HANDSHAKE = 0,
	WS_STATE_OPEN,
	WS_STATE_CLOSED,
} ws_state_t;

typedef enum {
	WS_REQUEST_INVALID = -1,
	WS_REQUEST_INCOMPLETE = 0,
	WS_REQUEST_WEBSOCKET = 1,
	WS_REQUEST_HTTP = 2,
} ws_request_kind_t;

int ws_classify_request(const uint8_t *rx_data, size_t len, char *sec_key_out,
			size_t sec_key_out_size);

int ws_build_handshake_response(const char *sec_key, uint8_t *tx_buf, size_t tx_buf_size);

int ws_build_http_response_headers(uint8_t *tx_buf, size_t tx_buf_size, const char *status,
				   const char *content_type, size_t body_len);

int ws_parse_frame(const uint8_t *raw, size_t raw_len, uint8_t *payload_out,
		   size_t payload_out_size, uint16_t *opcode_out);

size_t ws_build_frame(uint8_t *out, size_t out_size, uint8_t opcode, const uint8_t *payload,
		      size_t payload_len, bool mask);
