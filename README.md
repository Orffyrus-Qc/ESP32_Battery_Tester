# ESP32 Non-Rechargeable Battery Tester

A short-load battery tester for **non‑rechargeable** (primary) cells. You pick a
battery type on a 16x2 I2C LCD (or enter your own "Custom" cell), the ESP32 applies
a **brief** controlled load, measures voltage & current, and feeds those readings
into a tiny onboard "micro AI model" that scores the cell **0..100 %** with a label
(EXCELLENT / GOOD / FAIR / REJECTED).

The load is only applied for 3–6 seconds on purpose. Primary cells have a **single
cycle** — draining them for more than a few seconds would waste the very life you're
trying to test.

---

## What it does

1. **Select a battery type** up/down through a preset list (AA/AAA/C/D, 9V, Lithium,
   CR2032 …).
2. **Custom mode** — if none fit, enter the chemistry's **full‑charge open‑circuit
   voltage (V)** and **nominal current (A)** with up/down, confirm with select.
3. **Measure** the open‑circuit voltage with no load.
4. **Apply a short load** via a P‑MOSFET PWM driver and sample V & I for 3–6 s.
5. **Score** the cell with the on‑board model and show the result for ~3 s.
6. Loop back to the top for the next cell.

---

## Wiring

| Signal            | ESP32 pin | Notes |
|-------------------|-----------|-------|
| LCD SDA          | GPIO 21   | I2C (backpack 0x27) |
| LCD SCL          | GPIO 22   | |
| VOLTAGE (ADC)    | GPIO 35   | from a resistor divider over the cell |
| current (ADC)    | GPIO 34   | ACS712 (or 0–5 V shunt amp) OUT |
| LOAD PWM          | GPIO 26   | drives a P‑MOSFET high‑side switch |
| activity LED     | GPIO 2    | reflects the load state |
| button UP         | GPIO 5    | active‑low, pull‑up |
| button DOWN       | GPIO 18   | active‑low, pull‑up |
| button SELECT     | GPIO 19   | active‑low, pull‑up |

### Voltage divider (battery → GPIO 35)

```
battery+ ──R1──(MID)──R2── battery─
                  │
              GPIO35
```
`VOLTAGE_DIVIDER_RATIO = (R1+R2)/R2`. With 30 kΩ top / 30 kΩ bottom that ratio is
2.0 (matches `constants.h` → full cell V = ADC reading × 2).

### Current sensing

An ACS712 5 A board is easiest — its OUT pin (≈0 V at 0 A; this sketch zeroes the
quiescent mid‑point internally) goes to **GPIO 34**. A shunt + differential op‑amp
amplified to 0–5 V works too — adjust `read_load_current()` to match your scaling.

> **NOTE:** the ESP32 ADC is **not** isolated from the cell — a ground‑referenced
> divider/shunt connects the circuit's ground to the battery's negative. That is fine
> for bench testing of low‑voltage primaries.

---

## Files

| File              | Purpose |
|-------------------|---------|
| `ESP32_Battery_Tester.ino`        | Main sketch: menu, load driver, measurements, micro‑AI model |
| `constants.h`                     | Pin assignments + calibration constants |
| `platformio.ini`                  | Build config (PlatformIO) |
| `README.md`                       | This file |

## Build / flash — PlatformIO

```bash
pio run -t upload -m monitor        # upload + open the serial monitor
```

The model weights live in `ESP32_Battery_Tester.ino` (`ML_W`, `ML_OUT_W`). Re‑train by
editing those float tables.

## Build / flash — Arduino IDE

1. Install the **ESP32 board package** (via Boards Manager).
2. Library Manager → install **LiquidCrystal_I2C** (m5stack/Frank Bruch).
3. Pick **Board: ESP32 Dev Module**, correct port, then **Upload**.

Change the LCD address if your backpack differs (`0x27` ⇄ `0x3f`) in `constants.h`.

---

## Model notes

The "micro AI model" is a single‑hidden‑layer heuristic: three normalized features
(open‑circuit V, loaded V, loaded current vs. the entry) pass through one sigmoid
layer, then a weighted readout is clamped to 0..100 %. It is deliberately weight‑sparse
and deterministic — suitable for an ESP32 with no ML framework — and is a **heuristic
estimate**, not a laboratory‑grade calibration.

## Safety

- Primary cells can vent if forced hard. This uses only a **short, light** load.
- The circuit shares ground with the cell. Keep it to the low voltages shown here.
- Verify the current sensor range for whatever cell you test (9 V / lithium cells draw
  more headroom).
