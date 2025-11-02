import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ---------------------- CONFIG ----------------------
PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"
FEATURES_PATH = r"C:\Users\rick2\Documents\PPG Project\data\features"


# ---------------------- BEAT DETECTION ----------------------
def detect_beats(ppg, fs=100, plot=False):
    """Detect systolic peaks and foot points (valleys) in PPG signal."""

    # Normalize signal
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    ppg = ppg - np.mean(ppg)

    # Peak and valley detection
    min_distance = int(0.4 * fs)  # ~150 BPM max
    prominence = 0.3 * np.std(ppg)
    peaks, _ = find_peaks(ppg, distance=min_distance, prominence=prominence)
    valleys, _ = find_peaks(-ppg, distance=min_distance, prominence=prominence / 2)

    # Keep only valleys before peaks
    paired_feet = []
    for p in peaks:
        prior_feet = [v for v in valleys if v < p]
        if prior_feet:
            paired_feet.append(prior_feet[-1])
    feet = np.array(paired_feet)

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(ppg, color='black', linewidth=1.1, label="PPG (Normalized)")
        plt.scatter(peaks, ppg[peaks], color='red', label='Peaks')
        plt.scatter(feet, ppg[feet], color='blue', label='Feet')
        plt.title("Detected Beats (Feet & Peaks)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "ppg_norm": ppg,
        "peaks": peaks,
        "feet": feet,
        "beats_count": len(peaks)
    }


# ---------------------- FEATURE EXTRACTION ----------------------
def extract_ppg_features(ppg, fs, beats_info):
    """Extract HR, HRV, amplitude, rise time, and variability metrics."""

    peaks = beats_info["peaks"]
    feet = beats_info["feet"]
    ppg_norm = beats_info["ppg_norm"]

    if len(peaks) < 2:
        print("⚠️ Not enough beats detected for feature extraction.")
        return {}

    # Compute peak times
    peak_times = peaks / fs
    ibi = np.diff(peak_times)
    ibi_mean = np.mean(ibi)
    ibi_std = np.std(ibi)
    hr = 60.0 / ibi_mean if ibi_mean > 0 else 0.0
    hrv = ibi_std * 1000  # ms-based

    # Pair peaks with previous feet safely
    paired_feet = []
    paired_peaks = []
    for p in peaks:
        prior_feet = [f for f in feet if f < p]
        if prior_feet:
            paired_feet.append(prior_feet[-1])
            paired_peaks.append(p)

    paired_feet = np.array(paired_feet)
    paired_peaks = np.array(paired_peaks)

    if len(paired_peaks) > 0 and len(paired_feet) > 0:
        amps = ppg_norm[paired_peaks] - ppg_norm[paired_feet]
        amp_mean = np.mean(amps)
        amp_std = np.std(amps)

        rise_times = (paired_peaks - paired_feet) / fs
        rt_mean = np.mean(rise_times)
        rt_std = np.std(rise_times)
    else:
        amp_mean = amp_std = rt_mean = rt_std = 0.0

    return {
        "heart_rate_bpm": hr,
        "ibi_mean_s": ibi_mean,
        "ibi_std_s": ibi_std,
        "hrv_ms": hrv,
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "rise_time_mean_s": rt_mean,
        "rise_time_std_s": rt_std,
        "beats_detected": len(peaks)
    }


# ---------------------- FILE HANDLING ----------------------
def get_latest_processed_file():
    csv_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith("_processed.csv")]
    if not csv_files:
        print("❌ No processed files found.")
        return None
    latest = max([os.path.join(PROCESSED_PATH, f) for f in csv_files], key=os.path.getmtime)
    return latest


# ---------------------- MAIN PIPELINE ----------------------
def run_feature_extraction(plot=True, save=True):
    os.makedirs(FEATURES_PATH, exist_ok=True)

    latest_file = get_latest_processed_file()
    if latest_file is None:
        return

    print(f"📁 Using latest processed file: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    if "ir_filtered" not in df.columns:
        print("❌ Missing 'ir_filtered' column. Run preprocessing first.")
        return

    ppg = df["ir_filtered"].to_numpy()

    # Auto-detect sampling frequency
    if "timestamp_ms" in df.columns:
        timestamps = df["timestamp_ms"].to_numpy()
        fs = 1000.0 / np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 100
    else:
        fs = 100

    print(f"🕒 Sampling Frequency: {fs:.2f} Hz")

    beats_info = detect_beats(ppg, fs=fs, plot=plot)
    features = extract_ppg_features(ppg, fs, beats_info)

    if not features:
        print("⚠️ No features extracted.")
        return

    # --- Get metadata ---
    print("\nPlease enter basic metadata:")
    age = int(input("Age (years): ") or 0)
    sex = input("Sex (M/F): ").strip().upper() or "U"
    weight = float(input("Weight (kg): ") or 0)
    height = float(input("Height (cm): ") or 0)

    bmi = round(weight / ((height / 100) ** 2), 2) if weight > 0 and height > 0 else 0

    metadata = {
        "age": age,
        "sex": sex,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi
    }

    # --- Combine and save ---
    result = {**metadata, **features}
    df_out = pd.DataFrame([result])

    if save:
        out_path = os.path.join(
            FEATURES_PATH,
            os.path.basename(latest_file).replace("_processed.csv", "_features.csv")
        )
        df_out.to_csv(out_path, index=False)
        print(f"\n💾 Saved extracted features → {out_path}")

    # --- Clean and display neatly ---
    print("\n✅ Extracted Features:")
    df_pretty = df_out.T.reset_index()
    df_pretty.columns = ['Feature', 'Value']
    print(df_pretty.to_string(index=False))


# ---------------------- RUN ----------------------
if __name__ == "__main__":
    run_feature_extraction(plot=True, save=True)
