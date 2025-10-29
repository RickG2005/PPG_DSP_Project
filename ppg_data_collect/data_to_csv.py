import serial
import csv
import os
from datetime import datetime

# --- USER SETTINGS ---
PORT = "COM5"          # 🔧 Replace with your ESP32's serial port (e.g., "COM3" on Windows, "/dev/ttyUSB0" on Linux)
BAUD_RATE = 115200     # Must match Arduino Serial.begin(115200)
OUTPUT_DIR = "C:\Users\rick2\Documents\PPG Project\data\raw"  # Save in data/raw relative to this script
# ----------------------

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a timestamped filename for organization
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"ppg_data_{timestamp}.csv")

print(f"Connecting to {PORT} at {BAUD_RATE} baud...")
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

# Open file and begin logging
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms", "ir_value"])  # CSV header

    print(f"Logging data to: {output_file}")
    print("Press Ctrl + C to stop recording.\n")

    try:
        while True:
            line = ser.readline().decode("utf-8").strip()
            if line:
                # If ESP32 is outputting just IR values (no timestamp)
                if "," not in line:
                    writer.writerow([datetime.now().timestamp() * 1000, line])
                else:
                    # If timestamp and value both printed from ESP32
                    parts = line.split(",")
                    if len(parts) == 2:
                        writer.writerow(parts)
                f.flush()  # Save incrementally to file
                print(line)
    except KeyboardInterrupt:
        print("\n Logging stopped by user.")
    except Exception as e:
        print(f"Error: {e}")

ser.close()
print(f"Data saved successfully to: {output_file}")
