#include "tcp_transport.h"

#include <errno.h>

#include <zephyr/kernel.h>
#include <zephyr/net/socket.h>
#include <zephyr/sys/printk.h>

#ifndef TCP_RX_BUFFER_SIZE
#define TCP_RX_BUFFER_SIZE 2048
#endif

#ifndef TCP_RX_THREAD_STACK_SIZE
#define TCP_RX_THREAD_STACK_SIZE 6144
#endif

#ifndef TCP_RX_THREAD_PRIORITY
#define TCP_RX_THREAD_PRIORITY 6
#endif

static int server_sock = -1;
static int client_sock = -1;

static uint16_t listen_port;
static tcp_on_connect_cb_t cb_connect;
static tcp_on_data_cb_t cb_data;
static tcp_on_disconnect_cb_t cb_disconnect;
static void *cb_user_ctx;

static atomic_t running;

K_THREAD_STACK_DEFINE(rx_thread_stack, TCP_RX_THREAD_STACK_SIZE);
static struct k_thread rx_thread_data;
static k_tid_t rx_tid;

static int server_open(uint16_t port)
{
	if (server_sock >= 0) {
		zsock_close(server_sock);
		server_sock = -1;
	}

	server_sock = zsock_socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (server_sock < 0) {
		printk("tcp_transport: socket failed: %d\n", errno);
		return -errno;
	}

	struct sockaddr_in addr = {
		.sin_family = AF_INET,
		.sin_port = htons(port),
		.sin_addr.s_addr = INADDR_ANY,
	};

	if (zsock_bind(server_sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
		printk("tcp_transport: bind failed: %d\n", errno);
		zsock_close(server_sock);
		server_sock = -1;
		return -errno;
	}

	if (zsock_listen(server_sock, 1) != 0) {
		printk("tcp_transport: listen failed: %d\n", errno);
		zsock_close(server_sock);
		server_sock = -1;
		return -errno;
	}

	printk("tcp_transport: listening on port %u\n", port);
	return 0;
}

static void close_client(void)
{
	if (client_sock >= 0) {
		zsock_close(client_sock);
		client_sock = -1;
	}
}

void tcp_transport_disconnect_client(void)
{
	close_client();
}

static void rx_thread_fn(void *a, void *b, void *c)
{
	ARG_UNUSED(a);
	ARG_UNUSED(b);
	ARG_UNUSED(c);

	uint8_t rx_buffer[TCP_RX_BUFFER_SIZE];

	while (atomic_get(&running)) {
		struct sockaddr_in client_addr;
		socklen_t addrlen = sizeof(client_addr);

		printk("tcp_transport: waiting for client...\n");
		int cs = zsock_accept(server_sock, (struct sockaddr *)&client_addr, &addrlen);
		if (cs < 0) {
			if (!atomic_get(&running)) {
				break;
			}
			continue;
		}

		client_sock = cs;

		/* Evict silent clients: the host streams continuously (>= 30 Hz),
		 * so 5 s without a byte means a dead/ghost peer (e.g. a crashed
		 * host that never sent FIN). Without this, one stale connection
		 * blocks the accept loop forever and NEW clients can never join. */
		struct zsock_timeval rcvto = {.tv_sec = 5, .tv_usec = 0};

		zsock_setsockopt(client_sock, SOL_SOCKET, SO_RCVTIMEO, &rcvto, sizeof(rcvto));

		if (cb_connect) {
			cb_connect(cb_user_ctx);
		}

		while (atomic_get(&running)) {
			int n = zsock_recv(client_sock, rx_buffer, sizeof(rx_buffer), 0);
			if (n <= 0) {
				close_client();
				if (cb_disconnect) {
					cb_disconnect(cb_user_ctx);
				}
				break;
			}

			if (cb_data) {
				cb_data(rx_buffer, (size_t)n, cb_user_ctx);
			}
		}

		close_client();
	}

	close_client();
	if (server_sock >= 0) {
		zsock_close(server_sock);
		server_sock = -1;
	}
}

int tcp_transport_start(uint16_t port, tcp_on_connect_cb_t on_connect, tcp_on_data_cb_t on_data,
			tcp_on_disconnect_cb_t on_disconnect, void *user_ctx)
{
	if (atomic_get(&running)) {
		return -EALREADY;
	}

	listen_port = port;
	cb_connect = on_connect;
	cb_data = on_data;
	cb_disconnect = on_disconnect;
	cb_user_ctx = user_ctx;

	int rc = server_open(port);
	if (rc != 0) {
		return rc;
	}

	atomic_set(&running, 1);
	rx_tid = k_thread_create(&rx_thread_data, rx_thread_stack,
				 K_THREAD_STACK_SIZEOF(rx_thread_stack), rx_thread_fn, NULL,
				 NULL, NULL, TCP_RX_THREAD_PRIORITY, 0, K_NO_WAIT);
	k_thread_name_set(rx_tid, "tcp_rx");

	return 0;
}

void tcp_transport_stop(void)
{
	if (!atomic_get(&running)) {
		return;
	}

	atomic_set(&running, 0);
	close_client();

	if (server_sock >= 0) {
		zsock_close(server_sock);
		server_sock = -1;
	}

	if (rx_tid) {
		k_thread_abort(rx_tid);
		rx_tid = NULL;
	}
}

int tcp_transport_restart(void)
{
	if (!listen_port) {
		return -EINVAL;
	}

	tcp_transport_stop();
	return tcp_transport_start(listen_port, cb_connect, cb_data, cb_disconnect, cb_user_ctx);
}

int tcp_transport_send(const uint8_t *data, size_t len)
{
	if (client_sock < 0) {
		return -ENOTCONN;
	}

	int n = zsock_send(client_sock, data, len, 0);
	if (n < 0) {
		return -errno;
	}
	return n;
}

