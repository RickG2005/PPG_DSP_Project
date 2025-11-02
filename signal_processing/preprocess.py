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
    """Normalize & center PPG data with outlier clipping."""
    lower, upper = np.percentile(signal, [1, 99])
    signal = np.clip(signal, lower, upper)
    baseline = np.median(signal)
    signal_centered = signal - baseline
    std = np.std(signal_centered)
    if std < 1e-6:
        std = 1.0
    signal_normalized = signal_centered / std
    return signal_normalized


# ---------- MAIN PREPROCESS FUNCTION ----------
def run_preprocessing(input_file=None):
    """
    Phase 2: Preprocess PPG data.
    If no file is given, it automatically picks the latest from data/raw.
    Returns the output processed file path.
    """

    # Auto-select latest CSV if none provided
    if input_file is None:
        csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]
        if not csv_files:
            print("❌ No CSV files found in raw folder.")
            return None
        input_file = max([os.path.join(RAW_PATH, f) for f in csv_files], key=os.path.getctime)
        print(f"📁 Latest CSV detected: {os.path.basename(input_file)}")
    else:
        print(f"📥 Using input file: {os.path.basename(input_file)}")

    # --- Load and clean ---
    df = load_and_clean_csv(input_file)
    ir_signal = df["ir_value"].to_numpy()

    # --- Estimate Sampling Rate ---
    time_diff = np.diff(df["timestamp_ms"]) / 1000.0
    valid_diffs = time_diff[time_diff < np.percentile(time_diff, 95)]
    estimated_fs = round(1 / np.mean(valid_diffs)) if len(valid_diffs) > 0 else DEFAULT_FS
    print(f"🕒 Estimated Sampling Rate: {estimated_fs} Hz")

    # --- Remove startup samples ---
    skip_samples = int(estimated_fs * 2)
    if len(ir_signal) > skip_samples:
        ir_signal = ir_signal[skip_samples:]
        df = df.iloc[skip_samples:].reset_index(drop=True)

    # --- Calibrate and Filter ---
    ir_calibrated = auto_calibrate(ir_signal)
    lowcut = 0.5
    highcut = 4.0 if estimated_fs >= 50 else min(estimated_fs / 4, 3.0)
    order = 2 if len(ir_calibrated) < 500 else 3
    filtered = butter_bandpass_filter(ir_calibrated, lowcut, highcut, fs=estimated_fs, order=order)

    # --- Save processed CSV ---
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    output_file = os.path.join(PROCESSED_PATH, os.path.basename(input_file).replace(".csv", "_processed.csv"))
    df["ir_filtered"] = filtered
    df.to_csv(output_file, index=False)
    print(f"💾 Saved processed CSV: {output_file}")

    # --- Plot Preview ---
    plt.figure(figsize=(12, 4))
    plt.plot(ir_calibrated, label="Auto-Calibrated IR", color='gray', alpha=0.6)
    plt.plot(filtered, label="Filtered IR", color='blue', linewidth=1.5)
    plt.title(f"Preprocessed PPG Signal: {os.path.basename(input_file)}")
    plt.xlabel("Samples")
    plt.ylabel("Normalized IR Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return output_file


# ---------- RUN DIRECTLY ----------
if __name__ == "__main__":
    run_preprocessing()
