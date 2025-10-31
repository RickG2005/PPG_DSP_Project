import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ---------------------- CONFIG ----------------------
PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"
FS = 100  # Sampling frequency (Hz)


# ---------------------- CORE BEAT DETECTION ----------------------
def detect_beats(ppg, fs=100, plot=False):
    """
    Detect systolic peaks and foot points (onsets) in a PPG signal.
    """
    # Normalize signal
    ppg = (ppg - np.mean(ppg)) / np.std(ppg)

    # Detect systolic peaks (main upward peaks)
    min_distance = int(0.4 * fs)  # ~400 ms (≈150 bpm max)
    prominence = 0.4 * np.std(ppg)
    peaks, _ = find_peaks(ppg, distance=min_distance, prominence=prominence)

    # Detect feet (valleys before each peak)
    diff = np.gradient(ppg)
    feet, _ = find_peaks(-diff, distance=min_distance, prominence=0.2 * np.std(diff))

    foot_times = np.array(feet) / fs
    peak_times = np.array(peaks) / fs

    # --- Plot ---
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(ppg, label="PPG (Normalized)", color='black', linewidth=1.2)
        plt.scatter(peaks, ppg[peaks], color='red', label='Peaks', zorder=3)
        plt.scatter(feet, ppg[feet], color='blue', label='Feet', zorder=3)
        plt.title("Detected Beats (Feet & Peaks)")
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "foot_indices": feet.tolist(),
        "peak_indices": peaks.tolist(),
        "foot_times": foot_times.tolist(),
        "peak_times": peak_times.tolist(),
        "beats_count": len(peaks)
    }


# ---------------------- FIND LATEST FILE ----------------------
def get_latest_processed_file():
    """
    Finds the most recently modified processed CSV file.
    """
    csv_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith("_processed.csv")]
    if not csv_files:
        print("❌ No processed CSV files found in:", PROCESSED_PATH)
        return None
    latest_file = max(
        [os.path.join(PROCESSED_PATH, f) for f in csv_files],
        key=os.path.getmtime
    )
    return latest_file


# ---------------------- WRAPPER FUNCTION ----------------------
def run_latest_beat_detection(plot=True):
    """
    Automatically detects the latest processed CSV and runs beat detection.
    """
    latest_file = get_latest_processed_file()
    if latest_file is None:
        return None

    print(f"✅ Using latest processed file: {latest_file}")
    df = pd.read_csv(latest_file)

    if "ir_filtered" not in df.columns:
        print("❌ 'ir_filtered' column not found. Did you preprocess first?")
        return None

    ppg = df["ir_filtered"].values
    print(f"📈 Loaded {len(ppg)} samples for beat detection")

    results = detect_beats(ppg, fs=FS, plot=plot)
    print(f"✅ Detected {results['beats_count']} beats")

    return results


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    run_latest_beat_detection(plot=True)
