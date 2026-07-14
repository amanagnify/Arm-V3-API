#ifndef SERVO_H
#define SERVO_H

#include <stdint.h>

#define SERVO_CHANNELS 6

int servo_init(void);
int servo_set_pulse(uint8_t channel, uint16_t pulse_us);

#endif
