#include "app_callbacks.h"

#include <stdio.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "servo_control.h"
#include "tcp_transport.h"
#include "websocket_layer.h"

void handle_raw_packet(app_ctx_t *app, const uint8_t *data, size_t len);

/* Protocol V3 (FIRMWARE_CONTRACT.md rev 2): 19 bytes, XOR checksum, six
 * absolute servo angles (base|p1|p2|p3|roll|gripper). */
#define CONTROL_PACKET_MAGIC   0xA6U
#define CONTROL_PACKET_VERSION 0x03U
#define CONTROL_PACKET_LEN     19U
#define CONTROL_PACKET_MASK    0x3FU

static const char http_landing_page[] =
	"<!doctype html><html><head><meta charset=utf-8>"
	"<meta name=viewport content='width=device-width,initial-scale=1'>"
	"<title>Robot Arm V3</title>"
	"<style>body{font:16px sans-serif;margin:24px;background:#f4f7fb;color:#152033}"
	"button,input,select{font:inherit;margin:6px 0}label{display:block}#status{font-weight:700}"
	"</style></head><body><h1>Robot Arm V3</h1><p id=status>Disconnected</p>"
	"<button id=conn>Connect</button><button id=send disabled>Send packet</button>"
	"<label>Mode <select id=mode><option>A</option><option>H</option><option>M</option></select></label>"
	"<label>Base <input id=base type=range min=0 max=180 step=0.5 value=90></label>"
	"<label>P1 <input id=p1 type=range min=0 max=100 step=0.5 value=98></label>"
	"<label>P2 <input id=p2 type=range min=45 max=180 step=0.5 value=88></label>"
	"<label>P3 <input id=p3 type=range min=0 max=180 step=0.5 value=90></label>"
	"<label>Roll <input id=roll type=range min=0 max=180 step=0.5 value=90></label>"
	"<label>Gripper <input id=grip type=range min=36 max=166.5 step=0.5 value=166.5></label>"
	"<script>let ws,seq=0;const $=id=>document.getElementById(id),st=$('status'),btn=$('send');"
	"function q(v){v=Math.max(0,Math.min(180,Number(v)));return Math.round(v/180*65535);}"
	"function pkt(){const p=new Uint8Array(19),d=new DataView(p.buffer);p[0]=0xA6;p[1]=0x03;"
	"d.setUint16(2,seq=(seq+1)&65535);p[4]=$('mode').value.charCodeAt(0);p[5]=0x3F;"
	"const v=['base','p1','p2','p3','roll','grip'];"
	"for(let i=0;i<6;i++)d.setUint16(6+2*i,q($(v[i]).value));"
	"let c=0;for(let i=0;i<18;i++)c^=p[i];p[18]=c;return p;}"
	"$('conn').onclick=()=>{if(ws&&(ws.readyState===0||ws.readyState===1)){ws.close();return;}"
	"st.textContent='Connecting';ws=new WebSocket(`ws://${location.host}/`);ws.binaryType='arraybuffer';"
	"ws.onopen=()=>{st.textContent='Connected';btn.disabled=false;$('conn').textContent='Disconnect';};"
	"ws.onclose=()=>{st.textContent='Disconnected';btn.disabled=true;$('conn').textContent='Connect';};"
	"ws.onerror=()=>{st.textContent='Connection error';};};"
	"btn.onclick=()=>{if(ws&&ws.readyState===1)ws.send(pkt());};</script></body></html>";

static const char http_upgrade_hint[] =
	"Open http://<device-ip>:4210 in a browser, or use ws://<device-ip>:4210/ from a WebSocket client.\n";

static float qangle_to_float(uint16_t value)
{
	return ((float)value / 65535.0f) * 180.0f;
}

static bool packet_is_valid(const uint8_t *data, size_t len)
{
	if (len != CONTROL_PACKET_LEN)
	{
		return false;
	}
	if (data[0] != CONTROL_PACKET_MAGIC || data[1] != CONTROL_PACKET_VERSION)
	{
		return false;
	}
	char mode = (char)data[4];
	if (mode != 'A' && mode != 'H' && mode != 'M')
	{
		return false;
	}
	if (data[5] != CONTROL_PACKET_MASK)
	{
		return false;
	}

	uint8_t checksum = 0;
	for (size_t i = 0; i < CONTROL_PACKET_LEN - 1U; i++)
	{
		checksum ^= data[i];
	}
	if (checksum != data[CONTROL_PACKET_LEN - 1U])
	{
		return false;
	}
	return true;
}

static bool format_control_packet(const uint8_t *data, size_t len, char *buf, size_t buf_size)
{
	if (!packet_is_valid(data, len))
	{
		return false;
	}

	uint16_t sequence = ((uint16_t)data[2] << 8) | data[3];
	char mode = (char)data[4];
	float angle[6];

	for (int i = 0; i < 6; i++)
	{
		angle[i] = qangle_to_float(((uint16_t)data[6 + 2 * i] << 8) | data[7 + 2 * i]);
	}

	int written = snprintf(buf, buf_size,
			       "seq=%u mode=%c base=%.1f p1=%.1f p2=%.1f p3=%.1f roll=%.1f grip=%.1f",
			       (unsigned)sequence, mode, (double)angle[0], (double)angle[1],
			       (double)angle[2], (double)angle[3], (double)angle[4],
			       (double)angle[5]);
	return written > 0 && (size_t)written < buf_size;
}

static void drop_tcp_prefix(app_ctx_t *app, size_t shift)
{
	if (shift == 0)
	{
		return;
	}
	if (shift > app->tcp_raw_len)
	{
		shift = app->tcp_raw_len;
	}
	memmove(app->tcp_raw_buf, app->tcp_raw_buf + shift, app->tcp_raw_len - shift);
	app->tcp_raw_len -= shift;
}

static void drop_tcp_until_magic(app_ctx_t *app)
{
	size_t i = 0;

	while (i < app->tcp_raw_len && app->tcp_raw_buf[i] != CONTROL_PACKET_MAGIC)
	{
		i++;
	}

	if (i == app->tcp_raw_len)
	{
		app->tcp_raw_len = 0;
		return;
	}

	drop_tcp_prefix(app, i);
}

static void process_tcp_raw_stream(app_ctx_t *app, const uint8_t *data, size_t len)
{
	for (size_t n = 0; n < len; n++)
	{
		if (app->tcp_raw_len == sizeof(app->tcp_raw_buf))
		{
			drop_tcp_until_magic(app);
			if (app->tcp_raw_len == sizeof(app->tcp_raw_buf))
			{
				drop_tcp_prefix(app, 1);
			}
		}

		app->tcp_raw_buf[app->tcp_raw_len++] = data[n];

		while (app->tcp_raw_len >= CONTROL_PACKET_LEN)
		{
			drop_tcp_until_magic(app);
			if (app->tcp_raw_len < CONTROL_PACKET_LEN)
			{
				break;
			}

			if (packet_is_valid(app->tcp_raw_buf, CONTROL_PACKET_LEN))
			{
				handle_raw_packet(app, app->tcp_raw_buf, CONTROL_PACKET_LEN);
				drop_tcp_prefix(app, CONTROL_PACKET_LEN);
				continue;
			}

			/* Bad frame at a magic byte: advance to the next possible frame.
			 * Dropping at least one byte avoids livelock on a corrupt packet. */
			size_t shift = 1;
			while (shift < app->tcp_raw_len &&
			       app->tcp_raw_buf[shift] != CONTROL_PACKET_MAGIC)
			{
				shift++;
			}
			drop_tcp_prefix(app, shift);
		}
	}
}

void handle_raw_packet(app_ctx_t *app, const uint8_t *data, size_t len)
{
	k_mutex_lock(app->control_lock, K_FOREVER);
	bool ok = servo_control_apply_packet(data, (int)len);
	k_mutex_unlock(app->control_lock);

	if (!ok)
	{
		char packet_desc[128];

		if (format_control_packet(data, len, packet_desc, sizeof(packet_desc)))
		{
			printk("packet rejected: %s\n", packet_desc);
		}
		else
		{
			printk("packet rejected: invalid frame len=%u\n", (unsigned)len);
		}
	}
}

void app_on_tcp_connect(void *ctx)
{
	app_ctx_t *app = (app_ctx_t *)ctx;
	app->protocol_detected = false;
	app->use_websocket = false; // Will be detected on first data
	app->ws_state = WS_STATE_CLOSED;
	app->ws_raw_len = 0;
	app->tcp_raw_len = 0;
}

void app_on_tcp_disconnect(void *ctx)
{
	app_ctx_t *app = (app_ctx_t *)ctx;
	bool websocket_connected = app->use_websocket && app->ws_state == WS_STATE_OPEN;

	app->protocol_detected = false;
	app->use_websocket = false;
	app->ws_state = WS_STATE_CLOSED;
	app->ws_raw_len = 0;
	app->tcp_raw_len = 0;
	if (websocket_connected)
	{
		printk("WebSocket client disconnected\n");
	}
}

void app_on_tcp_data(const uint8_t *data, size_t len, void *ctx)
{
	app_ctx_t *app = (app_ctx_t *)ctx;
	char sec_key[80];
	bool data_already_buffered = false;

	sec_key[0] = '\0';

	if (!app->protocol_detected)
	{
		if (len > sizeof(app->ws_raw_buf) - app->ws_raw_len)
		{
			printk("tcp: pre-detect buf full, drop\n");
			tcp_transport_disconnect_client();
			return;
		}

		memcpy(&app->ws_raw_buf[app->ws_raw_len], data, len);
		app->ws_raw_len += len;
		data_already_buffered = true;

		/* WebSocket starts with "GET ". Raw V3 starts binary; tolerate a
		 * split HTTP request line before committing to raw TCP. */
		if (app->ws_raw_len < 4U && memcmp(app->ws_raw_buf, "GET ", app->ws_raw_len) == 0)
		{
			return;
		}

		if (app->ws_raw_len >= 4U && memcmp(app->ws_raw_buf, "GET ", 4) == 0)
		{
			app->use_websocket = true;
			app->ws_state = WS_STATE_HANDSHAKE;
			printk("ws: detected handshake\n");
		}
		else
		{
			app->use_websocket = false;
			printk("tcp: raw stream\n");
			process_tcp_raw_stream(app, app->ws_raw_buf, app->ws_raw_len);
			app->ws_raw_len = 0;
			app->protocol_detected = true;
			return;
		}
		app->protocol_detected = true;
	}

	if (!app->use_websocket)
	{
		process_tcp_raw_stream(app, data, len);
		return;
	}

	if (!data_already_buffered)
	{
		/* Check WebSocket buffer - if full, disconnect to free resources */
		if (len > sizeof(app->ws_raw_buf) - app->ws_raw_len)
		{
			printk("ws: buf full, drop\n");
			tcp_transport_disconnect_client();
			return;
		}

		memcpy(&app->ws_raw_buf[app->ws_raw_len], data, len);
		app->ws_raw_len += len;
	}

	if (app->ws_state == WS_STATE_HANDSHAKE)
	{
		int kind = ws_classify_request(app->ws_raw_buf, app->ws_raw_len, sec_key,
					       sizeof(sec_key));

		if (kind == WS_REQUEST_INCOMPLETE)
		{
			return;
		}

		if (kind == WS_REQUEST_WEBSOCKET)
		{
			int hs = ws_build_handshake_response(sec_key, app->ws_tx_buf,
							     sizeof(app->ws_tx_buf));

			if (hs < 0)
			{
				printk("ws: handshake response failed\n");
				tcp_transport_disconnect_client();
				return;
			}

			(void)tcp_transport_send(app->ws_tx_buf, (size_t)hs);
			app->ws_state = WS_STATE_OPEN;
			app->ws_raw_len = 0;
			printk("WebSocket client connected\n");
			return;
		}

		if (kind == WS_REQUEST_HTTP)
		{
			int hs = ws_build_http_response_headers(app->ws_tx_buf, sizeof(app->ws_tx_buf),
								"200 OK", "text/html; charset=UTF-8",
								sizeof(http_landing_page) - 1U);

			if (hs < 0)
			{
				printk("http: response header build failed\n");
				tcp_transport_disconnect_client();
				return;
			}

			(void)tcp_transport_send(app->ws_tx_buf, (size_t)hs);
			(void)tcp_transport_send((const uint8_t *)http_landing_page,
						 sizeof(http_landing_page) - 1U);
			tcp_transport_disconnect_client();
			return;
		}

		printk("ws: handshake failed\n");
		{
			int hs = ws_build_http_response_headers(app->ws_tx_buf, sizeof(app->ws_tx_buf),
								"426 Upgrade Required",
								"text/plain; charset=UTF-8",
								sizeof(http_upgrade_hint) - 1U);

			if (hs >= 0)
			{
				(void)tcp_transport_send(app->ws_tx_buf, (size_t)hs);
				(void)tcp_transport_send((const uint8_t *)http_upgrade_hint,
							 sizeof(http_upgrade_hint) - 1U);
			}
		}
		tcp_transport_disconnect_client();
		return;
	}

	if (app->ws_state != WS_STATE_OPEN)
	{
		return;
	}

	while (app->ws_raw_len > 0)
	{
		int pay = ws_parse_frame(app->ws_raw_buf, app->ws_raw_len, app->ws_payload_buf,
					 sizeof(app->ws_payload_buf), NULL);
		if (pay == 0)
		{
			return;
		}
		if (pay == -1)
		{
			tcp_transport_disconnect_client();
			return;
		}
		if (pay < 0)
		{
			app->ws_raw_len = 0;
			return;
		}

		size_t consumed = 2;
		uint8_t b1 = app->ws_raw_buf[1];
		uint64_t pl = (uint64_t)(b1 & 0x7F);
		if (pl == 126)
		{
			consumed += 2;
			pl = ((uint64_t)app->ws_raw_buf[2] << 8) | app->ws_raw_buf[3];
		}
		else if (pl == 127)
		{
			consumed += 8;
			pl = 0;
			for (int i = 0; i < 8; i++)
			{
				pl = (pl << 8) | app->ws_raw_buf[2 + i];
			}
		}
		if ((b1 & 0x80) != 0)
		{
			consumed += 4;
		}
		consumed += (size_t)pl;

		if (consumed > app->ws_raw_len)
		{
			app->ws_raw_len = 0;
			return;
		}

		handle_raw_packet(app, app->ws_payload_buf, (size_t)pay);

		memmove(app->ws_raw_buf, &app->ws_raw_buf[consumed], app->ws_raw_len - consumed);
		app->ws_raw_len -= consumed;
	}
}
