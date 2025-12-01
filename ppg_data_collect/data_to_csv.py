import serial
import csv
import os
import time
from datetime import datetime

# ---------------------- CONFIG ----------------------
PORT = "COM5"          # Adjust for your ESP32 port
BAUD_RATE = 115200     # Must match Arduino Serial.begin(115200)
RAW_PATH = r"C:\Users\rick2\Documents\PPG Project\data\raw"
WARMUP_SECONDS = 3     # Skip initial unstable readings
MIN_RECORDING_TIME = 30  # Minimum seconds for valid recording
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
        time.sleep(2)  # Wait for Arduino to initialize
    except Exception as e:
        print(f"❌ Could not open serial port {PORT}: {e}")
        print("💡 Tip: Check if Arduino IDE Serial Monitor is closed and PORT is correct.")
        return None

    # --- Warmup period ---
    print(f"⏳ Warming up sensor for {WARMUP_SECONDS} seconds...")
    warmup_start = time.time()
    while time.time() - warmup_start < WARMUP_SECONDS:
        ser.readline()  # Discard warmup readings
    print("✅ Warmup complete. Starting data collection...\n")

    # --- Start logging ---
    print(f"📄 Logging data to: {output_file}")
    print(f"⏱️  Minimum recording time: {MIN_RECORDING_TIME} seconds")
    print("🟢 Press Ctrl + C to stop recording.\n")

    start_time = time.time()
    sample_count = 0

    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "ir_value"])  # header

            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                
                if line:
                    # Skip any debug messages from Arduino
                    if "MAX30102" in line or "not found" in line:
                        print(f"ℹ️  {line}")
                        continue
                    
                    try:
                        # Handle ESP32 output: "timestamp,value"
                        if "," in line:
                            parts = line.split(",")
                            if len(parts) == 2:
                                ts, ir = parts
                                # Validate that both are numeric
                                int(ts)  # Check timestamp
                                int(ir)  # Check IR value
                                writer.writerow([ts, ir])
                                sample_count += 1
                                
                                # Show progress every 100 samples (~1 second at 100 Hz)
                                if sample_count % 100 == 0:
                                    elapsed = time.time() - start_time
                                    print(f"📊 Samples: {sample_count} | Time: {elapsed:.1f}s | Last IR: {ir}")
                        else:
                            # Handle single value (fallback)
                            int(line)  # Validate it's numeric
                            ts_ms = int(datetime.now().timestamp() * 1000)
                            writer.writerow([ts_ms, line])
                            sample_count += 1
                            
                    except ValueError:
                        # Skip non-numeric or malformed lines
                        continue
                    
                    f.flush()  # Save as we go

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n🛑 Logging stopped by user.")
        
        if elapsed < MIN_RECORDING_TIME:
            print(f"⚠️  Warning: Recording was only {elapsed:.1f}s (minimum recommended: {MIN_RECORDING_TIME}s)")
            print("   Short recordings may not have enough data for reliable analysis.")
        
    except Exception as e:
        print(f"⚠️ Error during data collection: {e}")
        return None
    finally:
        ser.close()

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n✅ Data saved successfully → {output_file}")
    print(f"📊 Total samples collected: {sample_count}")
    print(f"⏱️  Recording duration: {elapsed:.1f} seconds")
    print(f"📈 Average sampling rate: {sample_count/elapsed:.1f} Hz\n")
    
    # Check if we have enough data
    if sample_count < 100:
        print("⚠️  Very few samples collected. Please check sensor connection.")
        return None
    
    return output_file  # Return path to main.py for next phase


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    result = run_data_collection()
    if result:
        print(f"🎉 Ready to process: {result}")
    else:
        print("❌ Data collection failed.")