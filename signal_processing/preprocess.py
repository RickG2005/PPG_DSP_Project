import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RAW_PATH = r"C:\Users\rick2\Documents\PPG Project\data\raw"
PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"

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

    print("Looking for file at:", file_path)
    if not os.path.exists(file_path):
        print("❌ File not found — check RAW_PATH or filename!")
        return
    else:
        print("✅ File found, reading CSV...")

    
    df = pd.read_csv(file_path)
    df["time_s"] = df["timestamp_ms"] / 1000.0
    ir = df["ir"].values
    

    print(f"Loaded {len(df)} samples from {file_name}")
    print(f"IR min: {df['ir'].min()}, max: {df['ir'].max()}, mean: {df['ir'].mean():.2f}")

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
    fig, ax1 = plt.subplots(figsize=(10, 4))

    color = "tab:red"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Raw IR", color=color)
    ax1.plot(df["time_s"], ir, color=color, alpha=0.5, label="Raw IR")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # second y-axis for filtered signal
    color = "tab:blue"
    ax2.set_ylabel("Filtered (normalized)", color=color)
    ax2.plot(df["time_s"], ir_filtered, color=color, linewidth=2, label="Filtered")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("PPG Preprocessing: Raw vs Filtered")
    plt.show()



# ---------- RUN ----------
if __name__ == "__main__":
    # Run on your first collected CSV file
    preprocess_ppg("sample_ppg.csv")

