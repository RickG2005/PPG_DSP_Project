import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# ---------------------- CONFIG ----------------------
PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"


# ---------------------- BEAT DETECTION ----------------------
def detect_beats(ppg, fs=100, plot=False):
    """
    Detect systolic peaks and feet (valleys) in a PPG signal.

    Parameters:
        ppg : np.ndarray - filtered or clean PPG signal
        fs  : int or float - sampling frequency (Hz)
        plot: bool - show plot with peaks and feet
    Returns:
        dict containing peak/foot indices, times, and beat count
    """

    # --- Normalize properly ---
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    ppg = ppg - np.mean(ppg)

    # --- Detect peaks (systolic) ---
    min_distance = int(0.4 * fs)  # ~150 BPM max
    prominence = 0.3 * np.std(ppg)
    peaks, _ = find_peaks(ppg, distance=min_distance, prominence=prominence)

    # --- Detect valleys (feet) before peaks ---
    valleys, _ = find_peaks(-ppg, distance=min_distance, prominence=prominence / 2)

    # Keep only valleys that come before a peak
    paired_feet = []
    for p in peaks:
        prior_feet = [v for v in valleys if v < p]
        if prior_feet:
            paired_feet.append(prior_feet[-1])
    feet = np.array(paired_feet)

    # --- Convert indices to times ---
    foot_times = np.array(feet) / fs
    peak_times = np.array(peaks) / fs

    # --- Plot ---
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(ppg, color='black', linewidth=1.1, label="PPG (Normalized)")
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
    """Find the most recent processed CSV file."""
    csv_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith("_processed.csv")]
    if not csv_files:
        print("No processed CSV files found in:", PROCESSED_PATH)
        return None
    latest_file = max(
        [os.path.join(PROCESSED_PATH, f) for f in csv_files],
        key=os.path.getmtime
    )
    return latest_file


# ---------------------- WRAPPER FUNCTION ----------------------
def run_latest_beat_detection(plot=True):
    """Automatically load the latest processed CSV and detect beats."""

    latest_file = get_latest_processed_file()
    if latest_file is None:
        return None

    print(f"Using latest processed file: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    # --- Auto detect sampling frequency ---
    if "timestamp_ms" in df.columns:
        timestamps = df["timestamp_ms"].to_numpy()
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval_ms = np.mean(intervals)
            fs = 1000.0 / avg_interval_ms  # Hz
            print(f"Auto-detected sampling frequency: {fs:.2f} Hz")
        else:
            fs = 100
            print("Not enough data to auto-detect FS, defaulting to 100 Hz.")
    else:
        fs = 100
        print("No timestamp column found, defaulting to 100 Hz.")

    # --- Verify column ---
    if "ir_filtered" not in df.columns:
        print("'ir_filtered' column not found. Did you preprocess first?")
        return None

    ppg = df["ir_filtered"].to_numpy()
    print(f"Loaded {len(ppg)} samples for beat detection")

    results = detect_beats(ppg, fs=fs, plot=plot)
    print(f"Detected {results['beats_count']} beats")

    return results


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    run_latest_beat_detection(plot=True)
