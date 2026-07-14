#include "websocket_layer.h"

#include <string.h>

#include <zephyr/sys/printk.h>

typedef struct
{
	uint32_t h[5];
	uint64_t len_bits;
	uint8_t buf[64];
	size_t buf_len;
} sha1_ctx_t;

static uint32_t rol32(uint32_t x, uint32_t n)
{
	return (x << n) | (x >> (32U - n));
}

static void sha1_init(sha1_ctx_t *ctx)
{
	ctx->h[0] = 0x67452301U;
	ctx->h[1] = 0xEFCDAB89U;
	ctx->h[2] = 0x98BADCFEU;
	ctx->h[3] = 0x10325476U;
	ctx->h[4] = 0xC3D2E1F0U;
	ctx->len_bits = 0;
	ctx->buf_len = 0;
}

static void sha1_block(sha1_ctx_t *ctx, const uint8_t *block)
{
	uint32_t w[80];
	for (int i = 0; i < 16; i++)
	{
		w[i] = ((uint32_t)block[i * 4 + 0] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
			   ((uint32_t)block[i * 4 + 2] << 8) | ((uint32_t)block[i * 4 + 3]);
	}
	for (int i = 16; i < 80; i++)
	{
		w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
	}

	uint32_t a = ctx->h[0];
	uint32_t b = ctx->h[1];
	uint32_t c = ctx->h[2];
	uint32_t d = ctx->h[3];
	uint32_t e = ctx->h[4];

	for (int i = 0; i < 80; i++)
	{
		uint32_t f, k;
		if (i < 20)
		{
			f = (b & c) | ((~b) & d);
			k = 0x5A827999U;
		}
		else if (i < 40)
		{
			f = b ^ c ^ d;
			k = 0x6ED9EBA1U;
		}
		else if (i < 60)
		{
			f = (b & c) | (b & d) | (c & d);
			k = 0x8F1BBCDCU;
		}
		else
		{
			f = b ^ c ^ d;
			k = 0xCA62C1D6U;
		}

		uint32_t temp = rol32(a, 5) + f + e + k + w[i];
		e = d;
		d = c;
		c = rol32(b, 30);
		b = a;
		a = temp;
	}

	ctx->h[0] += a;
	ctx->h[1] += b;
	ctx->h[2] += c;
	ctx->h[3] += d;
	ctx->h[4] += e;
}

static void sha1_update(sha1_ctx_t *ctx, const uint8_t *data, size_t len)
{
	ctx->len_bits += (uint64_t)len * 8U;

	while (len > 0)
	{
		size_t to_copy = 64U - ctx->buf_len;
		if (to_copy > len)
		{
			to_copy = len;
		}

		memcpy(&ctx->buf[ctx->buf_len], data, to_copy);
		ctx->buf_len += to_copy;
		data += to_copy;
		len -= to_copy;

		if (ctx->buf_len == 64U)
		{
			sha1_block(ctx, ctx->buf);
			ctx->buf_len = 0;
		}
	}
}

static void sha1_final(sha1_ctx_t *ctx, uint8_t out20[20])
{
	ctx->buf[ctx->buf_len++] = 0x80;
	if (ctx->buf_len > 56U)
	{
		while (ctx->buf_len < 64U)
		{
			ctx->buf[ctx->buf_len++] = 0x00;
		}
		sha1_block(ctx, ctx->buf);
		ctx->buf_len = 0;
	}
	while (ctx->buf_len < 56U)
	{
		ctx->buf[ctx->buf_len++] = 0x00;
	}

	for (int i = 7; i >= 0; i--)
	{
		ctx->buf[ctx->buf_len++] = (uint8_t)((ctx->len_bits >> (i * 8)) & 0xFFU);
	}
	sha1_block(ctx, ctx->buf);

	for (int i = 0; i < 5; i++)
	{
		out20[i * 4 + 0] = (uint8_t)((ctx->h[i] >> 24) & 0xFFU);
		out20[i * 4 + 1] = (uint8_t)((ctx->h[i] >> 16) & 0xFFU);
		out20[i * 4 + 2] = (uint8_t)((ctx->h[i] >> 8) & 0xFFU);
		out20[i * 4 + 3] = (uint8_t)(ctx->h[i] & 0xFFU);
	}
}

static int base64_encode(const uint8_t *in, size_t in_len, char *out, size_t out_size)
{
	static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	size_t out_len = ((in_len + 2U) / 3U) * 4U;
	if (out_size < out_len + 1U)
	{
		return -1;
	}

	size_t oi = 0;
	for (size_t i = 0; i < in_len; i += 3U)
	{
		uint32_t v = 0;
		size_t rem = in_len - i;

		v |= (uint32_t)in[i + 0] << 16;
		if (rem > 1)
		{
			v |= (uint32_t)in[i + 1] << 8;
		}
		if (rem > 2)
		{
			v |= (uint32_t)in[i + 2];
		}

		out[oi++] = tbl[(v >> 18) & 0x3F];
		out[oi++] = tbl[(v >> 12) & 0x3F];
		out[oi++] = (rem > 1) ? tbl[(v >> 6) & 0x3F] : '=';
		out[oi++] = (rem > 2) ? tbl[v & 0x3F] : '=';
	}

	out[oi] = '\0';
	return (int)oi;
}

static const uint8_t ws_guid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

static char ascii_lower(char c)
{
	if (c >= 'A' && c <= 'Z')
	{
		return (char)(c + ('a' - 'A'));
	}
	return c;
}

static bool header_name_equal_span(const char *line, size_t line_len, const char *name,
				   const char **value_out, size_t *value_len_out)
{
	size_t name_len = strlen(name);

	if (line_len <= name_len || line[name_len] != ':')
	{
		return false;
	}

	for (size_t i = 0; i < name_len; i++)
	{
		if (ascii_lower(line[i]) != ascii_lower(name[i]))
		{
			return false;
		}
	}

	size_t value_start = name_len + 1U;
	while (value_start < line_len && (line[value_start] == ' ' || line[value_start] == '\t'))
	{
		value_start++;
	}

	size_t value_end = line_len;
	while (value_end > value_start &&
	       (line[value_end - 1U] == ' ' || line[value_end - 1U] == '\t'))
	{
		value_end--;
	}

	if (value_out)
	{
		*value_out = &line[value_start];
	}
	if (value_len_out)
	{
		*value_len_out = value_end - value_start;
	}

	return true;
}

static bool header_contains_token_span(const char *value, size_t value_len, const char *token)
{
	size_t token_len = strlen(token);

	if (token_len == 0 || value_len < token_len)
	{
		return false;
	}

	for (size_t i = 0; i + token_len <= value_len; i++)
	{
		bool match = true;

		for (size_t j = 0; j < token_len; j++)
		{
			if (ascii_lower(value[i + j]) != ascii_lower(token[j]))
			{
				match = false;
				break;
			}
		}

		if (match)
		{
			return true;
		}
	}

	return false;
}

static bool request_line_is_get(const char *line, size_t line_len)
{
	return line_len >= 14U && memcmp(line, "GET ", 4) == 0;
}

int ws_classify_request(const uint8_t *rx_data, size_t len, char *sec_key_out,
			size_t sec_key_out_size)
{
	const char *http = (const char *)rx_data;
	bool eoh = false;
	bool is_get = false;
	bool saw_upgrade_header = false;
	bool saw_connection_header = false;
	bool saw_key_header = false;
	bool saw_ws_hint = false;
	size_t parsed = 0;
	size_t line_index = 0;

	if (!sec_key_out || sec_key_out_size == 0U)
	{
		return WS_REQUEST_INVALID;
	}
	sec_key_out[0] = '\0';

	for (size_t i = 3; i < len; i++)
	{
		if (http[i - 3] == '\r' && http[i - 2] == '\n' &&
		    http[i - 1] == '\r' && http[i] == '\n')
		{
			eoh = true;
			break;
		}
	}
	if (!eoh)
	{
		return WS_REQUEST_INCOMPLETE;
	}

	while (parsed < len)
	{
		const char *line_start = http + parsed;
		const char *line_end = memchr(line_start, '\n', len - parsed);
		size_t line_len;
		const char *value = NULL;
		size_t value_len = 0U;

		if (!line_end)
		{
			break;
		}

		line_len = (size_t)(line_end - line_start);
		if (line_len > 0U && line_start[line_len - 1U] == '\r')
		{
			line_len--;
		}
		parsed = (size_t)(line_end - http) + 1U;

		if (line_len == 0U)
		{
			break;
		}

		if (line_index == 0U)
		{
			if (!request_line_is_get(line_start, line_len))
			{
				return WS_REQUEST_INVALID;
			}
			is_get = true;
			line_index++;
			continue;
		}

		if (header_name_equal_span(line_start, line_len, "Upgrade", &value, &value_len))
		{
			if (header_contains_token_span(value, value_len, "websocket"))
			{
				saw_ws_hint = true;
				saw_upgrade_header = true;
			}
		}
		else if (header_name_equal_span(line_start, line_len, "Connection", &value, &value_len))
		{
			if (header_contains_token_span(value, value_len, "upgrade"))
			{
				saw_ws_hint = true;
				saw_connection_header = true;
			}
		}
		else if (header_name_equal_span(line_start, line_len, "Sec-WebSocket-Key", &value,
						 &value_len))
		{
			saw_ws_hint = true;
			if (value_len == 0U || value_len >= sec_key_out_size)
			{
				return WS_REQUEST_INVALID;
			}
			memcpy(sec_key_out, value, value_len);
			sec_key_out[value_len] = '\0';
			saw_key_header = true;
		}
		else if (header_name_equal_span(line_start, line_len, "Sec-WebSocket-Version", NULL,
						 NULL))
		{
			saw_ws_hint = true;
		}

		line_index++;
	}

	if (!is_get)
	{
		return WS_REQUEST_INVALID;
	}

	if (saw_upgrade_header && saw_connection_header && saw_key_header)
	{
		return WS_REQUEST_WEBSOCKET;
	}

	if (saw_ws_hint)
	{
		printk("ws: invalid handshake headers\n");
		return WS_REQUEST_INVALID;
	}

	return WS_REQUEST_HTTP;
}

int ws_build_handshake_response(const char *sec_key, uint8_t *tx_buf, size_t tx_buf_size)
{
	sha1_ctx_t sha;
	uint8_t digest[20];
	char accept_b64[64];

	if (!sec_key || !tx_buf || tx_buf_size == 0U)
	{
		return -1;
	}

	sha1_init(&sha);
	sha1_update(&sha, (const uint8_t *)sec_key, strlen(sec_key));
	sha1_update(&sha, ws_guid, sizeof(ws_guid) - 1);
	sha1_final(&sha, digest);

	if (base64_encode(digest, sizeof(digest), accept_b64, sizeof(accept_b64)) < 0)
	{
		return -1;
	}

	int n = snprintk((char *)tx_buf, tx_buf_size,
			 "HTTP/1.1 101 Switching Protocols\r\n"
			 "Upgrade: websocket\r\n"
			 "Connection: Upgrade\r\n"
			 "Sec-WebSocket-Accept: %s\r\n"
			 "\r\n",
			 accept_b64);

	if (n < 0 || (size_t)n >= tx_buf_size)
	{
		return -1;
	}

	printk("ws: handshake success\n");
	return n;
}

int ws_build_http_response_headers(uint8_t *tx_buf, size_t tx_buf_size, const char *status,
				   const char *content_type, size_t body_len)
{
	int n;

	if (!tx_buf || !status || !content_type || tx_buf_size == 0U)
	{
		return -1;
	}

	n = snprintk((char *)tx_buf, tx_buf_size,
		     "HTTP/1.1 %s\r\n"
		     "Content-Type: %s\r\n"
		     "Cache-Control: no-store\r\n"
		     "Connection: close\r\n"
		     "Content-Length: %u\r\n"
		     "\r\n",
		     status, content_type, (unsigned int)body_len);

	if (n < 0 || (size_t)n >= tx_buf_size)
	{
		return -1;
	}

	return n;
}
int ws_parse_frame(const uint8_t *raw, size_t raw_len, uint8_t *payload_out,
				   size_t payload_out_size, uint16_t *opcode_out)
{
	if (raw_len < 2)
	{
		return 0;
	}

	uint8_t b0 = raw[0];
	uint8_t b1 = raw[1];
	uint8_t opcode = b0 & 0x0F;
	bool masked = (b1 & 0x80) != 0;
	uint64_t payload_len = (uint64_t)(b1 & 0x7F);
	size_t idx = 2;

	if ((b0 & 0x80) == 0)
	{
		return -2;
	}

	if (payload_len == 126)
	{
		if (raw_len < idx + 2)
		{
			return 0;
		}
		payload_len = ((uint64_t)raw[idx] << 8) | raw[idx + 1];
		idx += 2;
	}
	else if (payload_len == 127)
	{
		if (raw_len < idx + 8)
		{
			return 0;
		}
		payload_len = 0;
		for (int i = 0; i < 8; i++)
		{
			payload_len = (payload_len << 8) | raw[idx + i];
		}
		idx += 8;
	}

	uint8_t mask_key[4] = {0};
	if (masked)
	{
		if (raw_len < idx + 4)
		{
			return 0;
		}
		memcpy(mask_key, &raw[idx], 4);
		idx += 4;
	}

	if (payload_len > payload_out_size)
	{
		return -4;
	}
	if (payload_len > SIZE_MAX - idx)
	{
		return -4;
	}
	if (raw_len < idx + (size_t)payload_len)
	{
		return 0;
	}

	if (opcode_out)
	{
		*opcode_out = opcode;
	}

	if (opcode == 0x08)
	{
		return -1;
	}
	if (opcode != 0x02)
	{
		return -3;
	}

	for (uint64_t i = 0; i < payload_len; i++)
	{
		uint8_t v = raw[idx + (size_t)i];
		if (masked)
		{
			v ^= mask_key[i & 3U];
		}
		payload_out[i] = v;
	}

	return (int)payload_len;
}

size_t ws_build_frame(uint8_t *out, size_t out_size, uint8_t opcode, const uint8_t *payload,
					  size_t payload_len, bool mask)
{
	size_t idx = 0;
	if (out_size < 2)
	{
		return 0;
	}

	out[idx++] = 0x80 | (opcode & 0x0F);

	uint8_t maskbit = mask ? 0x80 : 0x00;
	if (payload_len <= 125U)
	{
		out[idx++] = maskbit | (uint8_t)payload_len;
	}
	else if (payload_len <= 0xFFFFU)
	{
		if (out_size < idx + 3)
		{
			return 0;
		}
		out[idx++] = maskbit | 126U;
		out[idx++] = (uint8_t)((payload_len >> 8) & 0xFFU);
		out[idx++] = (uint8_t)(payload_len & 0xFFU);
	}
	else
	{
		return 0;
	}

	uint8_t mask_key[4] = {0};
	if (mask)
	{
		if (out_size < idx + 4)
		{
			return 0;
		}
		mask_key[0] = 0x12;
		mask_key[1] = 0x34;
		mask_key[2] = 0x56;
		mask_key[3] = 0x78;
		memcpy(&out[idx], mask_key, 4);
		idx += 4;
	}

	if (out_size < idx + payload_len)
	{
		return 0;
	}

	for (size_t i = 0; i < payload_len; i++)
	{
		uint8_t v = payload[i];
		if (mask)
		{
			v ^= mask_key[i & 3U];
		}
		out[idx++] = v;
	}

	return idx;
}
