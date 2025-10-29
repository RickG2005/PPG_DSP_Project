## 📋 Hardware Requirements

- **ESP32 Development Board** (any variant with GPIO 21/22)
- **MAX30102 Pulse Oximeter Sensor Module**
- **USB Cable** (for power and data)
- **Jumper Wires** (4x Female-to-Female or Male-to-Female)

---

## 🔌 Pin Connections

### ESP32 ↔ MAX30102 Wiring

| MAX30102 Pin | ESP32 GPIO Pin | Description           |
|--------------|----------------|-----------------------|
| **VIN**      | **3.3V**       | Power (⚠️ NOT 5V!)   |
| **GND**      | **GND**        | Ground                |
| **SDA**      | **GPIO 21**    | I2C Data Line         |
| **SCL**      | **GPIO 22**    | I2C Clock Line        |

**Important Notes:**
- Use **3.3V only** - 5V will damage the sensor
- No external pull-up resistors needed (module has onboard resistors)
- I2C address: **0x57** (default, auto-detected)

---

## 💻 Software Setup

### 1. Arduino IDE Configuration

1. Install **Arduino IDE** (version 1.8.x or 2.x)
2. Add ESP32 board support:
   - File → Preferences → Additional Boards Manager URLs
   - Add: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install

3. Install **SparkFun MAX3010x Library**:
   - Sketch → Include Library → Manage Libraries
   - Search: "SparkFun MAX3010x"
   - Install: "SparkFun MAX3010x Pulse and Proximity Sensor Library"

### 2. Board Settings

- **Board:** "ESP32 Dev Module" (or your specific ESP32 variant)
- **Upload Speed:** 921600
- **CPU Frequency:** 240 MHz
- **Flash Frequency:** 80 MHz
- **Port:** Select your ESP32 COM port

---

## 📊 Sensor Configuration Parameters

### Working Configuration (Validated)

```cpp
Baud Rate:         115200
Sample Rate:       100 Hz (hardware)
Sample Averaging:  4 (effective rate = 25 Hz)
LED Mode:          2 (Red + IR, Red disabled)
LED Brightness:    0x1F (31/255)
Pulse Width:       411 μs
ADC Range:         16384 nA
IR LED Current:    0x1F
Red LED Current:   0x00 (disabled)
```