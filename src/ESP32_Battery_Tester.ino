/* =============================================================
   ESP32 Non-Rechargeable Battery Tester  (non-rechargeable branch)
===============================================================================
   Select a battery type on the 16x2 display, or choose "Custom" and type in the
   chemistry's full-charge open-circuit voltage AND its current rating. The
   ESP32 then performs a short load burst, samples voltage & current, and uses a
   tiny onboard "micro AI model" to score the cell 0 .. 100 (%).

   Load must be a brief burst. Primary cells have a single cycle — draining them
   for more than a few seconds would destroy the one use we are trying to test.

   Hardware / pin-outs live in constants.h
=============================================================================== */

#include <Wire.h>
#include "constants.h"

#include <LiquidCrystal_I2C.h>

// ───────────────────────── Module globals ─────────────────────────

static LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);

typedef enum {
    STATE_SELECT = 0,   // pick a type from the list, or "Custom"
    STATE_CUSTOM = 1,   // user types in V & current
    STATE_WAIT   = 2,   // settle + open-circuit measure
    STATE_LOAD   = 3,   // drive the load for the burst
    STATE_SCORE  = 4,   // compute the health score from the samples
    STATE_RESULT = 5,   // show the score, offer restart, or error
    STATE_ERROR  = 6    // sensor / config error
} menu_state;

static menu_state g_state = STATE_SELECT;

// selection state
static int      sel_index   = 0;
static bool     custom_mode = false;

// custom battery user input
static float    custom_open = CUSTOM_OPEN_V;   // full-charge open-circuit V
static float    custom_amp  = CUSTOM_DESIGN_A; // design/marked current (A)
static float    custom_res  = CUSTOM_DESIGN_R; // derived load resistance (Ω)

// test results
static float    ocv_v   = 0.0f;
static float    v_load  = 0.0f;
static float    i_load  = 0.0f;
static float    health_score = -1.0f;
static char     rating_label[RATING_BUF] = {0};

// debounce + custom-editor field selector (0 = voltage, 1 = current)
static uint32_t last_button_ms = 0;
static int      custom_edit_field = 0;

// ───────────────────────── Battery type database ─────────────────────────

typedef struct {
    const char *name;                // label in the list
    float       nominal_ocv;         // full-charge open-circuit V (new, resting)
    uint8_t     rated_current_mA;    // rated / expected sustainable current (mA)
} BatteryRow;

const BatteryRow g_db[BATTERY_DB_MAX_ENTRIES] = {
    {"AA Alkaline",  1.61f, 500},
    {"AAA Alkaline", 1.61f, 300},
    {"C Alkaline",   1.61f, 1000},
    {"D Alkaline",   1.61f, 1500},
    {"9V Alkaline",  9.40f, 100},
    {"AA Lithium",   1.80f, 800},
    {"CR2032 Coin",  3.30f,  50},
    {"Custom...",    0.00f,  0},      // sentinel row selects the custom menu
};

// ───────── Micro AI model (trained by tools/train_model.py) ─────────
// One tiny feed-forward net: 3 features -> hidden(3 sigmoid) -> scalar readout
// → 0..100 %. Each hidden unit sees one feature (element-wise):
//     z_k = F_k * W_k + B ;  h_k = sigmoid(-1.25*z_k);
//     acc = sum_k( h_k * OUT_W_k );  score = clamp(acc/3*100, 0, 100).
//   F0 = ocv / nominal_ocv      F1 = loaded_v / nominal_ocv
//   F2 = loaded_i / rated_current  (each ~= 1.0 means nominal / fresh)
// Re-train with:  python tools/train_model.py
static const float ML_W[ML_HIDDEN_LAYER_SIZE]   = {  1.4561f,  1.8524f,  1.9404f };
static const float ML_B                           = -1.3004f;
static const float ML_OUT_W[ML_HIDDEN_LAYER_SIZE] = {  1.5928f,  1.6003f,  1.5575f };





static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-1.25f * x)); }

static float micro_model_score(float f0, float f1, float f2) {
    const float in[ML_HIDDEN_LAYER_SIZE] = { f0, f1, f2 };
    float acc = 0.0f;
    for (int k = 0; k < ML_HIDDEN_LAYER_SIZE; ++k) {
        float h = sigmoid(in[k] * ML_W[k] + ML_B);
        acc = acc + h * ML_OUT_W[k];
    }
    float score = (acc / ML_HIDDEN_LAYER_SIZE) * HEALTH_MAX;   // 0..100 %
    return clampf(score, HEALTH_MIN, HEALTH_MAX);
}

static void build_label(float score, char *out) {
    if      (score >= 60.0f) strncpy(out, "EXCELLENT", RATING_BUF - 1);
    else if (score >= 40.0f) strncpy(out, "Good",      RATING_BUF - 1);
    else if (score >= 20.0f) strncpy(out, "Fair",      RATING_BUF - 1);
    else                     strncpy(out, "REJECTED",  RATING_BUF - 1);
    out[RATING_BUF - 1] = '\0';
}

// ───────────────────────── ADC + current helpers ─────────────────────────

static float read_battery_voltage() {
    int raw = analogRead(VOLTAGE_ADC_PIN);
    float v = ((float)raw / (float)ADC_RESOLUTION) * ADC_FULLSCALE_V;
    return v * VOLTAGE_DIVIDER_RATIO;
}

static float read_load_current() {
    int raw = analogRead(CURRENT_ADC_PIN);
    float v = ((float)raw / (float)ADC_RESOLUTION) * ADC_FULLSCALE_V;
    // ACS712: 1.5 V quiescent (0 A), 0.05 V per amp → amps = (V-1.5)/0.05
    float amps = (v - 1.5f) / 0.05f;
    return amps > 0.0f ? amps : 0.0f;
}

// ───────────────────────── Load driver (PWM) ─────────────────────────

static void load_pwm(float duty) {
    duty = clampf(duty, 0.0f, 1.0f);
    uint8_t value = (uint8_t)((float)(duty * 255.0f) + 0.5f);   // 8-bit resolution
    analogWrite(PIN_LOAD_PWM, value);                          // 1 kHz, 8-bit on the load channel
    digitalWrite(LED_PIN, value > 0 ? HIGH : LOW);             // activity LED reflects load
}

// ───────────────────────── Buttons (active-low, debounce) ─────────────────────────

static int read_buttons() {
    uint32_t now = millis();
    if (now - last_button_ms < 250) return NO_PRESS;
    last_button_ms = now;
    if      (digitalRead(BUTTON_UP_PIN)    == LOW) return PRESS_UP;
    else if (digitalRead(BUTTON_DOWN_PIN)  == LOW) return PRESS_DOWN;
    else if (digitalRead(BUTTON_SELECT_PIN)==LOW) return PRESS_SELECT;
    return NO_PRESS;
}

// ───────────────────────── Display helpers ─────────────────────────

static void draw_select() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("== BATTERY SELECT ==");
    lcd.setCursor(0, 1);
    char buf[20];
    snprintf(buf, sizeof(buf), "%2d %-13s", sel_index + 1, g_db[sel_index].name);
    lcd.print(buf);
}

static const char *sel_marker() {
    static char mark = ' ';
    mark = custom_edit_field == 0 ? '>' : ' ';       // highlight voltage line
    return &mark;
}
static const char *sel_marker_c() {
    static char mark = ' ';
    mark = custom_edit_field == 1 ? '>' : ' ';       // highlight current line
    return &mark;
}

static void draw_custom_edit() {
    lcd.clear();
    lcd.setCursor(0, 0);
    char vbuf[16];
    snprintf(vbuf, sizeof(vbuf), "%s%.2f V", sel_marker(), custom_open);
    lcd.print(vbuf);
    lcd.setCursor(0, 1);
    char abuf[16];
    snprintf(abuf, sizeof(abuf), "%s%.2f A", sel_marker_c(), custom_amp);
    lcd.print(abuf);
}

static void draw_result() {
    lcd.clear();
    lcd.setCursor(0, 0);
    char line[20];
    snprintf(line, sizeof(line), "%.0f %%", health_score);
    lcd.print(line);
    lcd.setCursor(0, 1);
    lcd.print(rating_label);
}

static void draw_error() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("ERROR:");
    lcd.setCursor(0, 1);
    lcd.print("RECHECK WIRING");
}

// ───────────────────────── Main loop ─────────────────────────

void loop() {
    int btn = read_buttons();

    switch (g_state) {
    case STATE_SELECT:
        draw_select();
        if      (btn == PRESS_UP)     { if (sel_index < BATTERY_DB_MAX_ENTRIES - 1) ++sel_index; }
        else if (btn == PRESS_DOWN)   { if (sel_index > 0) --sel_index; }
        else if (btn == PRESS_SELECT) {
            if (sel_index == BATTERY_DB_MAX_ENTRIES - 1) {
                custom_mode = true;
                custom_edit_field = 0;
                g_state      = STATE_CUSTOM;
            } else {
                custom_mode = false;
                g_state     = STATE_WAIT;
            }
        }
        break;

    case STATE_CUSTOM:
        draw_custom_edit();
        if      (btn == PRESS_UP)     { if (custom_edit_field == 0) custom_open  = clampf(custom_open  + 0.1f, 0.5f, 12.0f);
                                          else                  custom_amp  = clampf(custom_amp  + 0.01f, 0.01f, 5.0f); }
        else if (btn == PRESS_DOWN)   { if (custom_edit_field == 0) custom_open  = clampf(custom_open  - 0.1f, 0.5f, 12.0f);
                                          else                  custom_amp  = clampf(custom_amp  - 0.01f, 0.01f, 5.0f); }
        else if (btn == PRESS_SELECT) {
            if      (custom_edit_field == 0) custom_edit_field = 1;
            else {                                                        // moving off current → lock values, start test
                custom_res = (custom_open > 0.0f && custom_amp  > 0.0f) ? custom_open / custom_amp
                                                                        : CUSTOM_DESIGN_R;
                g_state = STATE_WAIT;
            }
        }
        break;

    case STATE_WAIT:
        delay(80);
        ocv_v = read_battery_voltage();
        if (ocv_v < VOLTAGE_MIN_V) { g_state = STATE_ERROR; break; }    // dead / missing cell
        g_state = STATE_LOAD;
        break;

    case STATE_LOAD: {
        int secs       = custom_mode ? TEST_LONG_SECONDS : TEST_SHORT_SECONDS;
        float designed = custom_mode ? custom_amp
                                        : (float)g_db[sel_index].rated_current_mA / 1000.0f;
        // 80 % duty gives a clear sag to read during a short burst (seconds)
        load_pwm(0.8f);
        for (int t = 0; t < secs; ++t) {
            delay(1000);
            v_load = read_battery_voltage();
            i_load = read_load_current();
        }
        load_pwm(0.0f);                            // cut the load
        (void)designed;                            // (used as reference for scoring)
        g_state = STATE_SCORE;
        break;
    }

    case STATE_SCORE: {
        float nominal_ocv = custom_mode ? custom_open : g_db[sel_index].nominal_ocv;
        float designed    = custom_mode ? custom_amp : (float)g_db[sel_index].rated_current_mA / 1000.0f;

        float f0 = nominal_ocv > 0.0f ? clampf(ocv_v / nominal_ocv, 0.0f, 1.6f) : 1.0f;
        float f1 = nominal_ocv > 0.0f ? clampf(v_load / nominal_ocv, 0.0f, 1.6f) : 1.0f;
        float f2 = designed > 0.0f  ? clampf(i_load / designed, 0.0f, 1.6f) : 1.0f;

        health_score = micro_model_score(f0, f1, f2);
        build_label(health_score, rating_label);
        g_state = STATE_RESULT;
        break;
    }

    case STATE_RESULT:
        draw_result();
        delay(3000);
        g_state = STATE_SELECT;
        break;

    case STATE_ERROR:
        draw_error();
        delay(1500);
        g_state = STATE_SELECT;
        break;
    }
}

// ───────────────────────── Setup ─────────────────────────

void setup() {
    lcd.init();
    lcd.backlight();
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    pinMode(BUTTON_UP_PIN,    INPUT_PULLUP);
    pinMode(BUTTON_DOWN_PIN,  INPUT_PULLUP);
    pinMode(BUTTON_SELECT_PIN,INPUT_PULLUP);
    pinMode(PIN_LOAD_PWM, OUTPUT);
    load_pwm(0.0f);                                // safe start: no load applied
    g_state = STATE_SELECT;
}
