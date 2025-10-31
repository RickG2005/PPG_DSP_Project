import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import re
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RAW_PATH = r"C:\Users\rick2\Documents\PPG Project\data\raw"
PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"
DEFAULT_FS = 100  # Default fallback sampling frequency


# ---------- CLEAN RAW CSV ----------
def load_and_clean_csv(file_path):
    clean_rows = []
    with open(file_path, "r") as f:
        for line in f:
            if not re.match(r"^\d+(\.\d+)?,\d+(\.\d+)?$", line.strip()):
                continue
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    timestamp = float(parts[0])
                    ir_value = float(parts[1])
                    clean_rows.append((timestamp, ir_value))
                except ValueError:
                    continue

    if not clean_rows:
        raise ValueError(f"No valid data found in {file_path}")

    df = pd.DataFrame(clean_rows, columns=["timestamp_ms", "ir_value"])
    return df


# ---------- FILTER ----------
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ---------- MAIN PREPROCESS ----------
def preprocess_latest_csv():
    # 1️⃣ Find latest CSV
    csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in raw folder.")
        return

    latest_file = max([os.path.join(RAW_PATH, f) for f in csv_files], key=os.path.getctime)
    print(f"📁 Latest CSV detected: {os.path.basename(latest_file)}")

    # 2️⃣ Load and clean
    df = load_and_clean_csv(latest_file)
    ir_signal = df["ir_value"].to_numpy()

    # 3️⃣ Estimate Sampling Rate (auto-detect)
    time_diff = np.diff(df["timestamp_ms"]) / 1000.0  # ms → s
    valid_diffs = time_diff[time_diff < np.percentile(time_diff, 95)]  # remove outliers
    if len(valid_diffs) > 0:
        estimated_fs = round(1 / np.mean(valid_diffs))
    else:
        estimated_fs = DEFAULT_FS

    print(f"🕒 Estimated Sampling Rate: {estimated_fs} Hz")

    # 4️⃣ Normalize
    ir_signal = (ir_signal - np.min(ir_signal)) / (np.max(ir_signal) - np.min(ir_signal))

    # 5️⃣ Adaptive Filter Settings
    lowcut = 0.5
    highcut = 4.0 if estimated_fs >= 50 else min(estimated_fs / 4, 3.0)
    order = 2 if len(ir_signal) < 500 else 3

    filtered = butter_bandpass_filter(ir_signal, lowcut, highcut, fs=estimated_fs, order=order)

    # 6️⃣ Edge Trimming (reduce filtfilt boundary distortion)
    trim = min(50, len(filtered) // 10)
    trimmed_raw = ir_signal[trim:-trim]
    trimmed_filtered = filtered[trim:-trim]

    # 7️⃣ Save processed version
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    out_file = os.path.join(PROCESSED_PATH, os.path.basename(latest_file).replace(".csv", "_processed.csv"))
    df["ir_filtered"] = filtered
    df.to_csv(out_file, index=False)
    print(f"💾 Saved processed CSV: {out_file}")
    print(f"✅ Processed {len(df)} samples (trimmed view: {len(trimmed_filtered)})")

    # 8️⃣ Plot (Clean vs Filtered only)
    plt.figure(figsize=(12, 4))
    raw_norm = (trimmed_raw - np.mean(trimmed_raw)) / np.std(trimmed_raw)
    filt_norm = (trimmed_filtered - np.mean(trimmed_filtered)) / np.std(trimmed_filtered)

    plt.plot(raw_norm, label="Raw IR (Normalized)", color='gray', alpha=0.6)
    plt.plot(filt_norm, label="Filtered IR (Normalized)", color='blue', linewidth=1.5)
    plt.title(f"Cleaned & Filtered PPG Signal: {os.path.basename(latest_file)}")
    plt.xlabel("Samples (trimmed)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- RUN ----------
if __name__ == "__main__":
    preprocess_latest_csv()
