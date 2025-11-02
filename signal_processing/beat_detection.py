import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"


# ---------- BEAT DETECTION CORE ----------
def detect_beats(ppg, fs=100, plot=False):
    """
    Detect systolic peaks and feet (valleys) in a PPG signal.
    Returns dictionary with indices, times, and beat count.
    """
    # --- Normalize ---
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    ppg = ppg - np.mean(ppg)

    # --- Detect peaks ---
    min_distance = int(0.4 * fs)  # Avoid >150 BPM
    prominence = 0.3 * np.std(ppg)
    peaks, _ = find_peaks(ppg, distance=min_distance, prominence=prominence)

    # --- Detect valleys before peaks ---
    valleys, _ = find_peaks(-ppg, distance=min_distance, prominence=prominence / 2)
    feet = []
    for p in peaks:
        prior_feet = [v for v in valleys if v < p]
        if prior_feet:
            feet.append(prior_feet[-1])
    feet = np.array(feet)

    # --- Convert to time ---
    foot_times = np.array(feet) / fs
    peak_times = np.array(peaks) / fs

    # --- Plot ---
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(ppg, color='black', linewidth=1.1, label="PPG (Normalized)")
        plt.scatter(peaks, ppg[peaks], color='red', label='Peaks')
        plt.scatter(feet, ppg[feet], color='blue', label='Feet')
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


# ---------- GET LATEST PROCESSED FILE ----------
def get_latest_processed_file():
    csv_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith("_processed.csv")]
    if not csv_files:
        print("❌ No processed CSV files found in:", PROCESSED_PATH)
        return None
    latest_file = max([os.path.join(PROCESSED_PATH, f) for f in csv_files], key=os.path.getmtime)
    return latest_file


# ---------- MAIN FUNCTION ----------
def run_latest_beat_detection(plot=True):
    """
    Load latest processed PPG data and run beat detection.
    Returns dictionary with detected peaks/feet info (used by feature_extraction.py).
    """
    latest_file = get_latest_processed_file()
    if latest_file is None:
        return None

    print(f"\n📁 Latest Processed File: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    # --- Auto detect sampling rate ---
    if "timestamp_ms" in df.columns:
        timestamps = df["timestamp_ms"].to_numpy()
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval_ms = np.mean(intervals)
            fs = 1000.0 / avg_interval_ms
        else:
            fs = 100
    else:
        fs = 100
    print(f"🕒 Sampling Rate: {fs:.2f} Hz")

    # --- Verify input ---
    if "ir_filtered" not in df.columns:
        print("❌ 'ir_filtered' column missing. Run preprocess.py first.")
        return None

    ppg = df["ir_filtered"].to_numpy()
    print(f"📈 Loaded {len(ppg)} samples for beat detection")

    # --- Run detection ---
    results = detect_beats(ppg, fs=fs, plot=plot)
    print(f"✅ Detected {results['beats_count']} beats")

    return results


# ---------- RUN DIRECTLY ----------
if __name__ == "__main__":
    run_latest_beat_detection(plot=True)
