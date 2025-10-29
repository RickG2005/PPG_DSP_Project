import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RAW_PATH = "C:\Users\rick2\Documents\PPG Project\data\raw"
PROCESSED_PATH = "C:\Users\rick2\Documents\PPG Project\data\processed"
FS = 100  # Sampling frequency (Hz) — same as ESP32 delay (10ms = 100Hz)


# ---------- FILTER ----------
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal, lowcut=0.5, highcut=8):
    b, a = butter_bandpass(lowcut, highcut, FS)
    return filtfilt(b, a, signal)


# ---------- MAIN PREPROCESS ----------
def preprocess_ppg(file_name):
    # 1️. Read data
    file_path = os.path.join(RAW_PATH, file_name)
    df = pd.read_csv(file_path)
    df["time_s"] = df["timestamp_ms"] / 1000.0
    ir = df["ir"].values

    # 2️. Filter and normalize
    ir_filtered = bandpass_filter(ir)
    ir_filtered = (ir_filtered - np.mean(ir_filtered)) / np.std(ir_filtered)

    # 3️. Save
    processed_df = pd.DataFrame({
        "time_s": df["time_s"],
        "ir_raw": ir,
        "ir_filtered": ir_filtered
    })

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    out_file = os.path.join(PROCESSED_PATH, file_name.replace(".csv", "_processed.csv"))
    processed_df.to_csv(out_file, index=False)
    print(f"Saved processed file: {out_file}")

    # 4. Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], ir, label="Raw", alpha=0.6)
    plt.plot(df["time_s"], ir_filtered, label="Filtered", linewidth=2)
    plt.title("PPG Preprocessing")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- RUN ----------
if __name__ == "__main__":
    # Run on your first collected CSV file
    preprocess_ppg("session1.csv")
