#pragma once

#include <stddef.h>
#include <stdint.h>

typedef void (*tcp_on_data_cb_t)(const uint8_t *data, size_t len, void *user_ctx);
typedef void (*tcp_on_connect_cb_t)(void *user_ctx);
typedef void (*tcp_on_disconnect_cb_t)(void *user_ctx);

int tcp_transport_start(uint16_t port, tcp_on_connect_cb_t on_connect, tcp_on_data_cb_t on_data,
			tcp_on_disconnect_cb_t on_disconnect, void *user_ctx);
void tcp_transport_stop(void);
int tcp_transport_restart(void);

int tcp_transport_send(const uint8_t *data, size_t len);

/* Force-close current client (keeps server running and returns to accept) */
void tcp_transport_disconnect_client(void);

