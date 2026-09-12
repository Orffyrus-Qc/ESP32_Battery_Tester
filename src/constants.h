/* =============================================================
   ESP32 Non-Rechargeable Battery Tester — Pinout & Constants
   ================================================================ */

#ifndef BATTERY_TESTER_CONSTANTS_H
#define BATTERY_TESTER_CONSTANTS_H

// ────────────── LCD ──────────────
#define LCD_I2C_ADDR        0x27          // PCF8574 backpack address (some units use 0x3f)
#define LCD_COLS            16
#define LCD_ROWS            2

// ────────────── Pins ──────────────
#define PIN_SDA             21          // I2C SDA (default ESP32)
#define PIN_SCL             22          // I2C SCL (default ESP32)

// ADC sensors
#define VOLTAGE_ADC_PIN     35          // ADC1_CH7 — battery voltage via divider (~3.9V span, EXT_11dB)
#define CURRENT_ADC_PIN     34          // ACS712 / shunt amplifier output

// Control + input
#define PIN_LOAD_PWM        26          // P‑MOSFET load via PWM 1 kHz, 8‑bit duty
#define LED_PIN             2          // built‑in status LED
#define BUTTON_UP_PIN       5
#define BUTTON_DOWN_PIN     18
#define BUTTON_SELECT_PIN   19

// ────────────── ADC ──────────────
#define ADC_FULLSCALE_V     3.865f        // ADC1 12-bit + ATTEN_11dB
#define ADC_RESOLUTION      4095          // 12-bit ADC max

// ────────────── Voltage divider (battery+ → R1 → MID → R2 → battery‑) ──────────────
// MID node → ADC. RATIO = (R1+R2)/R2
#define VOLTAGE_DIVIDER_RATIO 2.0f      // 30kΩ top / 30kΩ bottom → scales ADC back to cell volts
#define VOLTAGE_MIN_V       0.6f          // below this a primary cell is essentially dead

// ────────────── Buttons ──────────────
#define NO_PRESS            0
#define PRESS_DOWN          1
#define PRESS_UP            2
#define PRESS_SELECT        3

// ────────────── Databases ──────────────
#define BATTERY_DB_MAX_ENTRIES 8          // 7 real types + "Custom" sentinel
#define ML_HIDDEN_LAYER_SIZE  3           // neurons in the tiny model

// ────────────── Measurement windows (s) — short test to preserve cell life ──────────────
#define TEST_SHORT_SECONDS  3            // burst for a normal cell
#define TEST_LONG_SECONDS   6            // longer burst for the "Custom deep" profile

// ────────────── Custom / design specs ──────────────
#define CUSTOM_OPEN_V       1.6f          // assumed full‑charge open‑circuit V for a custom cell (AA)
#define CUSTOM_DESIGN_A     0.10f          // assumed max current (A) the custom cell can deliver
#define CUSTOM_DESIGN_R     15.0f          // R = V/I for the nominal load (Ω)

// ────────────── Health thresholds / labels ──────────────
#define HEALTH_MIN      0.0f
#define HEALTH_MAX      100.0f
#define RATING_BUF      16

#endif /* BATTERY_TESTER_CONSTANTS_H */
