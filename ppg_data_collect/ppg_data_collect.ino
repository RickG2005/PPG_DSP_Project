#include <Wire.h>
#include "MAX30105.h"       
#include "heartRate.h"     

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Check wiring and power.");
    while (1);
  }

  // --- CONFIGURATION ---
  byte ledBrightness = 0x1F;   // IR LED brightness (31 = low)
  byte sampleAverage = 1;      // Use 1 for true 100 Hz timing (avoid extra averaging)
  byte ledMode = 2;            // 2 = IR + Red (you can still keep Red OFF)
  int sampleRate = 100;        // Sampling frequency = 100 samples/sec
  int pulseWidth = 411;        // Good balance between resolution and range
  int adcRange = 16384;        // Wider ADC range to prevent saturation

  // --- Initialize the MAX30102 ---
  particleSensor.setup(
    ledBrightness,
    sampleAverage,
    ledMode,
    sampleRate,
    pulseWidth,
    adcRange
  );

  // Turn OFF Red LED (only use IR)
  particleSensor.setPulseAmplitudeRed(0x00);
  particleSensor.setPulseAmplitudeIR(0x1F);

  Serial.println("MAX30102 initialized at 100 Hz.");
  delay(1000);
}

void loop() {
  static unsigned long lastPrint = 0;
  long irValue = particleSensor.getIR();
  unsigned long timestamp = millis();

  // Only print new data when a new sample is available
  if (millis() - lastPrint >= 10) {  // ~100 Hz = every 10 ms
    Serial.print(timestamp);
    Serial.print(",");
    Serial.println(irValue);
    lastPrint = millis();
  }
}
