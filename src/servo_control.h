#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

int servo_control_init(void);
bool servo_control_apply_packet(const uint8_t *buf, int len);
void servo_control_update_outputs(void);
void servo_control_print_teleop_state(void);
uint32_t servo_control_last_sequence(void);
char servo_control_last_mode(void);


#endif
