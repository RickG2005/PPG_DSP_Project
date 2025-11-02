import serial
import csv
import os
from datetime import datetime

# ---------------------- CONFIG ----------------------
PORT = "COM5"          # Adjust for your ESP32 port
BAUD_RATE = 115200     # Must match Arduino Serial.begin(115200)
RAW_PATH = r"C:\Users\rick2\Documents\PPG Project\data\raw"
# ----------------------------------------------------

def run_data_collection():
    """
    Phase 1: Collects PPG data from serial and saves it as CSV.
    Automatically uses user's name in the filename.
    """
    # --- Ask for user name ---
    user_name = input("\n👤 Enter participant name: ").strip().replace(" ", "_")
    if not user_name:
        user_name = "unknown_user"

    # --- Prepare file path ---
    os.makedirs(RAW_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RAW_PATH, f"{user_name}_ppg_{timestamp}.csv")

    # --- Serial connection ---
    print(f"\n🔌 Connecting to {PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"❌ Could not open serial port {PORT}: {e}")
        return None

    # --- Start logging ---
    print(f"📄 Logging data to: {output_file}")
    print("🟢 Press Ctrl + C to stop recording.\n")

    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "ir_value"])  # header

            while True:
                line = ser.readline().decode("utf-8").strip()
                if line:
                    # Handle ESP32 output: either "value" or "timestamp,value"
                    if "," not in line:
                        writer.writerow([datetime.now().timestamp() * 1000, line])
                    else:
                        parts = line.split(",")
                        if len(parts) == 2:
                            writer.writerow(parts)

                    f.flush()  # save as we go
                    print(line)

    except KeyboardInterrupt:
        print("\n🛑 Logging stopped by user.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        ser.close()

    print(f"\n✅ Data saved successfully → {output_file}")
    return output_file  # <-- Return path to main.py for next phase


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    run_data_collection()
