#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found!");
    while (1);
  }
  
  // REDUCED LED current to prevent saturation
  byte ledBrightness = 0x1F;  // LOW brightness (31)
  byte sampleAverage = 4;
  byte ledMode = 2;
  int sampleRate = 100;
  int pulseWidth = 411;
  int adcRange = 16384;  // INCREASED range to prevent saturation
  
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, 
                       sampleRate, pulseWidth, adcRange);
  
  particleSensor.setPulseAmplitudeRed(0x00);   // Turn OFF Red
  particleSensor.setPulseAmplitudeIR(0x1F);    // LOW IR (start conservative)
}

void loop() {
  long irValue = particleSensor.getIR();
  Serial.println(irValue);
}