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

# ---------- AUTO CALIBRATION ----------
def auto_calibrate(signal):
    """
    Automatically scales and centers PPG data so that:
    - Outliers are clipped
    - Baseline drift is removed
    - Amplitude is normalized to a stable range
    """
    # 1️⃣ Clip extreme outliers
    lower, upper = np.percentile(signal, [1, 99])
    signal = np.clip(signal, lower, upper)

    # 2️⃣ Remove DC offset / baseline drift
    baseline = np.median(signal)
    signal_centered = signal - baseline

    # 3️⃣ Normalize amplitude dynamically
    std = np.std(signal_centered)
    if std < 1e-6:
        std = 1.0
    signal_normalized = signal_centered / std

    return signal_normalized

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

    # 4️⃣ Remove startup samples (sensor stabilization)
    skip_samples = int(estimated_fs * 2)
    if len(ir_signal) > skip_samples:
        ir_signal = ir_signal[skip_samples:]
        df = df.iloc[skip_samples:].reset_index(drop=True)

    # 5️⃣ Auto-calibrate signal
    ir_calibrated = auto_calibrate(ir_signal)

    # 6️⃣ Adaptive Filter
    lowcut = 0.5
    highcut = 4.0 if estimated_fs >= 50 else min(estimated_fs / 4, 3.0)
    order = 2 if len(ir_calibrated) < 500 else 3
    filtered = butter_bandpass_filter(ir_calibrated, lowcut, highcut, fs=estimated_fs, order=order)

    # 7️⃣ Edge trimming
    trim = min(50, len(filtered) // 10)
    trimmed_raw = ir_calibrated[trim:-trim]
    trimmed_filtered = filtered[trim:-trim]

    # 8️⃣ Save processed version
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    out_file = os.path.join(PROCESSED_PATH, os.path.basename(latest_file).replace(".csv", "_processed.csv"))
    df["ir_filtered"] = filtered
    df.to_csv(out_file, index=False)
    print(f"💾 Saved processed CSV: {out_file}")
    print(f"✅ Processed {len(df)} samples (trimmed view: {len(trimmed_filtered)})")

    # 9️⃣ Plot (Calibrated vs Filtered)
    plt.figure(figsize=(12, 4))
    plt.plot(trimmed_raw, label="Auto-Calibrated IR", color='gray', alpha=0.6)
    plt.plot(trimmed_filtered, label="Filtered IR", color='blue', linewidth=1.5)
    plt.title(f"Auto-Calibrated & Filtered PPG: {os.path.basename(latest_file)}")
    plt.xlabel("Samples")
    plt.ylabel("Normalized IR Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------- RUN ----------
if __name__ == "__main__":
    preprocess_latest_csv()
