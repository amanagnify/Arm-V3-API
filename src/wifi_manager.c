#include "wifi_manager.h"

#include <errno.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/net/net_event.h>
#include <zephyr/net/net_if.h>
#include <zephyr/net/net_ip.h>
#include <zephyr/net/net_mgmt.h>
#include <zephyr/net/wifi.h>
#include <zephyr/net/wifi_mgmt.h>
#include <zephyr/sys/printk.h>

#if defined(__has_include)
#if __has_include("wifi_credentials.local.h")
#include "wifi_credentials.local.h"
#endif
#endif

/* Defaults for public builds; override with src/wifi_credentials.local.h. */
#ifndef WIFI_MGR_SSID
#define WIFI_MGR_SSID "SET_WIFI_SSID"
#endif

#ifndef WIFI_MGR_PSK
#define WIFI_MGR_PSK "SET_WIFI_PASSWORD"
#endif

#ifndef WIFI_MGR_CHANNEL
#define WIFI_MGR_CHANNEL 6
#endif

#ifndef WIFI_MGR_TIMEOUT
#define WIFI_MGR_TIMEOUT K_SECONDS(10)
#endif

static struct net_if *wifi_iface;
static struct net_mgmt_event_callback net_mgmt_cb;
static K_SEM_DEFINE(ip_ready_sem, 0, 1);
static bool initialized;

static bool iface_has_ipv4(struct net_if *iface)
{
	return net_if_ipv4_get_global_addr(iface, NET_ADDR_PREFERRED) != NULL;
}

static void net_event_handler(struct net_mgmt_event_callback *cb, uint64_t mgmt_event,
							  struct net_if *iface)
{
	ARG_UNUSED(cb);

	if (iface != wifi_iface)
	{
		return;
	}

	if (mgmt_event == NET_EVENT_IPV4_ADDR_ADD)
	{
		k_sem_give(&ip_ready_sem);
	}
}

static void log_ipv4_address(struct net_if *iface)
{
	char ipbuf[40] = {0};
	struct net_in_addr *addr = net_if_ipv4_get_global_addr(iface, NET_ADDR_PREFERRED);

	if (addr != NULL)
	{
		net_addr_ntop(AF_INET, addr, ipbuf, sizeof(ipbuf));
		printk("ESP32 IP: %s\n", ipbuf);
		return;
	}

	printk("Preferred IPv4 not found\n");
}

int wifi_manager_init(void)
{
	if (initialized)
	{
		return 0;
	}

	wifi_iface = net_if_get_wifi_sta();
	if (!wifi_iface)
	{
		printk("wifi_manager: WiFi STA interface not found\n");
		return -ENODEV;
	}

	net_mgmt_init_event_callback(&net_mgmt_cb, net_event_handler, NET_EVENT_IPV4_ADDR_ADD);
	net_mgmt_add_event_callback(&net_mgmt_cb);

	initialized = true;
	return 0;
}

bool wifi_manager_is_connected(void)
{
	if (!initialized || !wifi_iface)
	{
		return false;
	}

	return iface_has_ipv4(wifi_iface);
}

int wifi_manager_connect(void)
{
	if (!initialized)
	{
		int rc = wifi_manager_init();
		if (rc != 0)
		{
			return rc;
		}
	}

	if (wifi_manager_is_connected())
	{
		return 0;
	}

	struct wifi_connect_req_params cnx = {0};
	cnx.ssid = WIFI_MGR_SSID;
	cnx.ssid_length = strlen(WIFI_MGR_SSID);
	cnx.psk = WIFI_MGR_PSK;
	cnx.psk_length = strlen(WIFI_MGR_PSK);
	cnx.security = WIFI_SECURITY_TYPE_PSK;
	/* IMPORTANT: Skip scan */
	// cnx.channel = WIFI_MGR_CHANNEL;
	cnx.channel = WIFI_CHANNEL_ANY;
	cnx.band = WIFI_FREQ_BAND_2_4_GHZ;

	(void)k_sem_take(&ip_ready_sem, K_NO_WAIT);

	printk("wifi_manager: Connecting WiFi...\n");

	int rc = net_mgmt(NET_REQUEST_WIFI_CONNECT, wifi_iface, &cnx, sizeof(cnx));
	if (rc != 0)
	{
		printk("wifi_manager: WiFi connect failed: %d\n", rc);
		return -EIO;
	}

	if (k_sem_take(&ip_ready_sem, WIFI_MGR_TIMEOUT) == 0)
	{
		printk("wifi_manager: WiFi connected\n");
		log_ipv4_address(wifi_iface);
		return 0;
	}

	printk("wifi_manager: Timeout waiting for IP\n");
	return -ETIMEDOUT;
}

int wifi_manager_reconnect_if_needed(void)
{
	if (wifi_manager_is_connected())
	{
		return 0;
	}

	return wifi_manager_connect();
}
