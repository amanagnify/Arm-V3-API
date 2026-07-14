#pragma once

#include <stdbool.h>

int wifi_manager_init(void);
int wifi_manager_connect(void); /* blocks until IP or timeout */
bool wifi_manager_is_connected(void);

/* Optional helper for heartbeat loops */
int wifi_manager_reconnect_if_needed(void);

