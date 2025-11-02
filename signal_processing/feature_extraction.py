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
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    ppg = ppg - np.mean(ppg)

    min_distance = int(0.4 * fs)
    prominence = 0.3 * np.std(ppg)
    peaks, _ = find_peaks(ppg, distance=min_distance, prominence=prominence)
    valleys, _ = find_peaks(-ppg, distance=min_distance, prominence=prominence / 2)

    paired_feet = []
    for p in peaks:
        prior_feet = [v for v in valleys if v < p]
        if prior_feet:
            paired_feet.append(prior_feet[-1])
    feet = np.array(paired_feet)

    foot_times = np.array(feet) / fs
    peak_times = np.array(peaks) / fs

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(ppg, color='black', linewidth=1.1, label="PPG (Normalized)")
        plt.scatter(peaks, ppg[peaks], color='red', label='Peaks')
        plt.scatter(feet, ppg[feet], color='blue', label='Feet')
        plt.title("Detected Beats")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "foot_indices": feet.tolist(),
        "peak_indices": peaks.tolist(),
        "foot_times": foot_times.tolist(),
        "peak_times": peak_times.tolist(),
        "beats_count": len(peaks)
    }


# ---------------------- METADATA INPUT ----------------------
def get_metadata():
    print("\nEnter metadata for this recording:")
    age = int(input("Age: "))
    sex = input("Sex (M/F/Other): ").strip().upper()
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))

    bmi = round(weight / ((height / 100) ** 2), 2)

    metadata = {
        "age": age,
        "sex": sex,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi
    }
    return metadata


# ---------------------- FEATURE EXTRACTION ----------------------
def extract_ppg_features(ppg, fs, beats_info):
    peaks = np.array(beats_info["peak_indices"])
    feet = np.array(beats_info["foot_indices"])

    min_len = min(len(peaks), len(feet))
    peaks, feet = peaks[:min_len], feet[:min_len]

    if len(peaks) < 2:
        print("Not enough peaks detected.")
        return {}

    # --- Core temporal features ---
    ibi = np.diff(beats_info["peak_times"])
    heart_rate = 60 / ibi
    hrv = np.std(ibi) * 1000

    # --- Normalize ---
    ppg_norm = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    amps = ppg_norm[peaks] - np.array([ppg_norm[f] for f in feet])
    rise_times = (peaks - feet) / fs

    # --- Morphological & vascular features ---
    sdr_values, pw50_values, auc_values, dt_ratios = [], [], [], []
    for i in range(len(peaks) - 1):
        start, peak, end = feet[i], peaks[i], feet[i + 1]
        segment = ppg_norm[start:end]

        # systolic/diastolic ratio
        if len(segment) > 2:
            mid_point = np.argmax(segment)
            systolic = segment[mid_point]
            diastolic = np.min(segment[mid_point:])
            if diastolic != 0:
                sdr_values.append(systolic / abs(diastolic))

        # Pulse width at half max
        half_max = (ppg_norm[peaks[i]] + ppg_norm[feet[i]]) / 2
        above_half = np.where(segment > half_max)[0]
        if len(above_half) > 1:
            pw50_values.append((above_half[-1] - above_half[0]) / fs)

        # Area under curve (AUC)
        auc_values.append(np.trapz(segment, dx=1 / fs))

        # Diastolic time ratio
        total_cycle = end - start
        diastolic_time = end - peak
        if total_cycle != 0:
            dt_ratios.append(diastolic_time / total_cycle)

    perfusion_index = (np.mean(amps) / np.max(ppg_norm)) * 100

    features = {
        # --- Time-domain ---
        "heart_rate_bpm": np.mean(heart_rate),
        "ibi_mean_s": np.mean(ibi),
        "ibi_std_s": np.std(ibi),
        "hrv_ms": hrv,

        # --- Amplitude & morphology ---
        "amplitude_mean": np.mean(amps),
        "amplitude_std": np.std(amps),
        "rise_time_mean_s": np.mean(rise_times),
        "rise_time_std_s": np.std(rise_times),

        # --- New vascular features ---
        "sdr_mean": np.mean(sdr_values) if sdr_values else np.nan,
        "pw50_mean_s": np.mean(pw50_values) if pw50_values else np.nan,
        "auc_mean": np.mean(auc_values) if auc_values else np.nan,
        "dt_ratio_mean": np.mean(dt_ratios) if dt_ratios else np.nan,
        "perfusion_index": perfusion_index,

        # --- Metadata about recording ---
        "beats_detected": len(peaks),
        "recording_duration_s": len(ppg) / fs
    }
    return features


# ---------------------- SANITY CHECK ----------------------
def sanity_check(metadata, features):
    warnings = []
    age = metadata.get("age")
    sex = metadata.get("sex", "").upper()
    weight = metadata.get("weight_kg")
    height = metadata.get("height_cm")
    bmi = metadata.get("bmi")

    hr = features.get("heart_rate_bpm")
    hrv = features.get("hrv_ms")
    ibi_mean = features.get("ibi_mean_s")
    amp_mean = features.get("amplitude_mean")
    rise_time = features.get("rise_time_mean_s")

    if age is not None and not (5 <= age <= 100):
        warnings.append(f"⚠️ Unrealistic age: {age}")
    if weight is not None and not (20 <= weight <= 200):
        warnings.append(f"⚠️ Unusual weight: {weight} kg")
    if height is not None and not (100 <= height <= 220):
        warnings.append(f"⚠️ Unusual height: {height} cm")
    if bmi is not None and not (10 <= bmi <= 40):
        warnings.append(f"⚠️ Out-of-range BMI: {bmi:.2f}")
    if sex not in ("M", "F", "OTHER"):
        warnings.append(f"⚠️ Unexpected sex value: '{sex}'")

    if hr is not None and not (40 <= hr <= 180):
        warnings.append(f"⚠️ Implausible heart rate: {hr:.1f} bpm")
    if ibi_mean is not None and hr is not None:
        hr_calc = 60 / ibi_mean
        if abs(hr - hr_calc) > 10:
            warnings.append(f"⚠️ HR/IBI mismatch: HR={hr:.1f}, IBI→HR={hr_calc:.1f}")
    if hrv is not None and not (10 <= hrv <= 300):
        warnings.append(f"⚠️ HRV looks abnormal: {hrv:.1f} ms")
    if amp_mean is not None and not (0.05 <= amp_mean <= 2.0):
        warnings.append(f"⚠️ PPG amplitude unusually low/high: {amp_mean:.3f}")
    if rise_time is not None and not (0.1 <= rise_time <= 0.6):
        warnings.append(f"⚠️ Rise time outside physiological range: {rise_time:.3f}s")

    return warnings


# ---------------------- FILE HANDLING ----------------------
def get_latest_processed_file():
    csv_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith("_processed.csv")]
    if not csv_files:
        print("No processed CSV files found.")
        return None
    latest_file = max([os.path.join(PROCESSED_PATH, f) for f in csv_files], key=os.path.getmtime)
    return latest_file


# ---------------------- MAIN RUN FUNCTION ----------------------
def run_feature_extraction(plot=False, save=True):
    latest_file = get_latest_processed_file()
    if latest_file is None:
        return

    print(f"\n📂 Using latest processed file: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    if "ir_filtered" in df.columns:
        ppg = df["ir_filtered"].to_numpy()
    elif "ppg" in df.columns:
        ppg = df["ppg"].to_numpy()
    else:
        print("No valid PPG column found.")
        return

    fs = 100
    beats_info = detect_beats(ppg, fs, plot=plot)
    metadata = get_metadata()
    features = extract_ppg_features(ppg, fs, beats_info)

    # Combine + sanity check
    result = {**metadata, **features}
    df_out = pd.DataFrame([result])

    warnings = sanity_check(metadata, features)
    if warnings:
        print("\n⚠️ Sanity Check Warnings:")
        for w in warnings:
            print("   " + w)
    else:
        print("\n✅ All extracted values look physiologically valid.")

    if save:
        out_path = os.path.join(
            FEATURES_PATH,
            os.path.basename(latest_file).replace("_processed.csv", "_features.csv")
        )
        df_out.to_csv(out_path, index=False)
        print(f"\n💾 Saved extracted features → {out_path}")

    print("\n✅ Extracted Features:")
    print(df_out.T)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    run_feature_extraction(plot=True)
