#include "servo.h"

#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <string.h>

/* ================= CONFIG ================= */

#define I2C0_NODE DT_NODELABEL(i2c0)
#define PCA9685_ADDR 0x40

#define MODE1 0x00
#define MODE2 0x01
#define PRESCALE 0xFE
#define LED0_ON_L 0x06

#define SERVO_FREQ_HZ 300.0f
#define SERVO_MIN_PULSE_US 500.0f
#define SERVO_MAX_PULSE_US 2500.0f
#define I2C_RETRY 3

/* ================= GLOBAL ================= */

static const struct device *i2c_dev = DEVICE_DT_GET(I2C0_NODE);
static uint16_t last_pwm[16]; // jitter reduction

/* ================= LOW LEVEL ================= */

static int write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};

    for (int i = 0; i < I2C_RETRY; i++)
    {
        if (i2c_write(i2c_dev, buf, 2, PCA9685_ADDR) == 0)
            return 0;

        k_usleep(500);
    }

    return -1;
}

static int read_reg(uint8_t reg, uint8_t *val)
{
    for (int i = 0; i < I2C_RETRY; i++)
    {
        if (i2c_write_read(i2c_dev, PCA9685_ADDR, &reg, 1, val, 1) == 0)
            return 0;

        k_usleep(500);
    }

    return -1;
}

static int pca9685_probe(void)
{
    uint8_t val;
    return read_reg(MODE1, &val);
}

/* ================= PCA9685 ================= */

static int set_pwm(uint8_t ch, uint16_t val)
{
    if (ch > 15)
        return -1;

    /* Skip identical value to reduce I2C traffic */
    if (last_pwm[ch] == val)
        return 0;

    uint8_t reg = LED0_ON_L + (4 * ch);

    uint8_t buf[5] =
    {
        reg,

        /* ON = 0 */
        0x00,
        0x00,

        /* OFF */
        (uint8_t)(val & 0xFF),
        (uint8_t)((val >> 8) & 0x0F)
    };

    for (int i = 0; i < I2C_RETRY; i++)
    {
        if (i2c_write(i2c_dev, buf, sizeof(buf), PCA9685_ADDR) == 0)
        {
            last_pwm[ch] = val;
            return 0;
        }

        k_usleep(500);
    }

    return -1;
}

/* ================= INIT ================= */
int servo_init(void)
{
    if (!device_is_ready(i2c_dev))
    {
        printk("I2C not ready\n");
        return -1;
    }

    if (pca9685_probe() != 0)
    {
        printk("PCA9685 not detected\n");
        return -1;
    }

    printk("PCA9685 detected");

    /* Reset */
    if (write_reg(MODE1, 0x00) != 0)
        return -1;
    k_msleep(10);

    /* Enable Auto Increment */
    if (write_reg(MODE1, 0x20) != 0)
        return -1;
    k_msleep(10);

    /* Output driver */
    if (write_reg(MODE2, 0x04) != 0)
        return -1;

    /* ---------- Frequency Setup ---------- */

    float freq = SERVO_FREQ_HZ;

    if (freq < 24.0f)
        freq = 24.0f;

    if (freq > 1526.0f)
        freq = 1526.0f;

    uint8_t prescale =
        (uint8_t)(25000000.0f /
                  (4096.0f * freq) -
                  1.0f +
                  0.5f);

    uint8_t oldmode;

    if (read_reg(MODE1, &oldmode) != 0)
        return -1;

    /* Sleep */
    uint8_t sleepmode =
        (oldmode & ~0x80) |
        0x10;

    if (write_reg(MODE1, sleepmode) != 0)
        return -1;

    if (write_reg(PRESCALE, prescale) != 0)
        return -1;

    k_msleep(5);

    /* Wake + Auto Increment */

    uint8_t wakemode =
        (oldmode & ~0x10) |
        0x20;

    if (write_reg(MODE1, wakemode) != 0)
        return -1;

    k_msleep(5);

    /* Restart */

    if (write_reg(MODE1, wakemode | 0x80) != 0)
        return -1;

    memset(last_pwm, 0xFF, sizeof(last_pwm));

    printk("PCA9685 initialized (%.1f Hz)\n",
           (double)SERVO_FREQ_HZ);

    return 0;
}

/* ================= HIGH LEVEL ================= */

int servo_set_pulse(uint8_t channel, uint16_t pulse_us)
{
    if (channel > 15U)
        return -1;

    if (pulse_us < (uint16_t)SERVO_MIN_PULSE_US)
        pulse_us = (uint16_t)SERVO_MIN_PULSE_US;
    if (pulse_us > (uint16_t)SERVO_MAX_PULSE_US)
        pulse_us = (uint16_t)SERVO_MAX_PULSE_US;

    float tick = ((float)pulse_us * SERVO_FREQ_HZ * 4096.0f) / 1000000.0f;
    uint16_t pwm = (uint16_t)(tick + 0.5f);

    if (pwm > 4095)
        pwm = 4095;

    return set_pwm(channel, pwm);
}
