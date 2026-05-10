import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROCESSED_PATH = r"C:\Users\rick2\Documents\PPG Project\data\processed"
FEATURES_PATH = r"C:\Users\rick2\Documents\PPG Project\data\features"


# Take metadata
def get_metadata():
    print("\n📋 Enter participant metadata and lifestyle details:")
    print("---------------------------------------------------")

    # Basic Demographics 
    age = int(input("Age (in years): "))
    sex = input("Sex (M/F/Other): ").strip().upper()
    weight = float(input("Weight (in kg): "))
    height = float(input("Height (in cm): "))

    # Lifestyle & Health Descriptors
    print("\n💤 Average sleep per night (in hours):")
    print("   - Try to give your regular nightly average.")
    sleep_hours = float(input("Hours of sleep: "))

    print("\n🏃 Physical Activity Level:")
    print("   - Sedentary: Little to no exercise (desk job, minimal movement)")
    print("   - Moderate: Exercise 2–3 times/week or walk often")
    print("   - Active: Regular workouts (4+ times/week, sports, gym)")
    activity_level = input("Choose (Sedentary / Moderate / Active): ").strip().lower()

    print("\n🩺 Immediate family diabetes history:")
    print("   - Type 'yes' if parents or siblings have diabetes.")
    family_diabetes = input("Immediate family has diabetes? (yes/no): ").strip().lower()

    print("\n🚬 Smoking habit:")
    print("   - Type 'yes' if you currently smoke or recently quit (<1 year).")
    smoker = input("Do you smoke? (yes/no): ").strip().lower()

    print("\n☕ Daily caffeine intake:")
    print("   - none: 0 cups/day")
    print("   - low: 1–2 cups (tea/coffee/energy drinks)")
    print("   - moderate: 3–4 cups")
    print("   - high: 5+ cups or energy drinks daily")
    caffeine_intake = input("Caffeine intake (none / low / moderate / high): ").strip().lower()

    # Derived Metrics 
    bmi = round(weight / ((height / 100) ** 2), 2)

    # Numeric encoding for easier modeling
    activity_score = {
        "sedentary": 1,
        "moderate": 2,
        "active": 3
    }.get(activity_level, np.nan)

    # Final Dictionary 
    metadata = {
        "age": age,
        "sex": sex,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi,
        "sleep_hours": sleep_hours,
        "activity_level": activity_level,
        "activity_score": activity_score,
        "family_diabetes": family_diabetes,
        "smoker": smoker,
        "caffeine_intake": caffeine_intake
    }

    print("\n✅ Recorded Metadata Summary:")
    print("---------------------------------------------------")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    return metadata



# Feature extraction
def extract_ppg_features(ppg, fs, beats_info):
    peaks = np.array(beats_info["peak_indices"])
    feet = np.array(beats_info["foot_indices"])

    min_len = min(len(peaks), len(feet))
    peaks, feet = peaks[:min_len], feet[:min_len]

    if len(peaks) < 2:
        print("⚠️ Not enough peaks detected.")
        return {}

    # Temporal features
    ibi = np.diff(beats_info["peak_times"])
    heart_rate = 60 / ibi
    hrv = np.std(ibi) * 1000

    # Normalize PPG 
    ppg_norm = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    amps = ppg_norm[peaks] - np.array([ppg_norm[f] for f in feet])
    rise_times = (peaks - feet) / fs

    # Morphological features 
    sdr_values, pw50_values, auc_values, dt_ratios = [], [], [], []
    for i in range(len(peaks) - 1):
        start, peak, end = feet[i], peaks[i], feet[i + 1]
        segment = ppg_norm[start:end]

        if len(segment) > 2:
            mid_point = np.argmax(segment)
            systolic = segment[mid_point]
            diastolic = np.min(segment[mid_point:])
            if diastolic != 0:
                sdr_values.append(systolic / abs(diastolic))

        half_max = (ppg_norm[peaks[i]] + ppg_norm[feet[i]]) / 2
        above_half = np.where(segment > half_max)[0]
        if len(above_half) > 1:
            pw50_values.append((above_half[-1] - above_half[0]) / fs)

        auc_values.append(np.trapz(segment, dx=1 / fs))
        total_cycle = end - start
        diastolic_time = end - peak
        if total_cycle != 0:
            dt_ratios.append(diastolic_time / total_cycle)

    perfusion_index = (np.mean(amps) / np.max(ppg_norm)) * 100

    features = {
        "heart_rate_bpm": np.mean(heart_rate),
        "ibi_mean_s": np.mean(ibi),
        "ibi_std_s": np.std(ibi),
        "hrv_ms": hrv,
        "amplitude_mean": np.mean(amps),
        "amplitude_std": np.std(amps),
        "rise_time_mean_s": np.mean(rise_times),
        "rise_time_std_s": np.std(rise_times),
        "sdr_mean": np.mean(sdr_values) if sdr_values else np.nan,
        "pw50_mean_s": np.mean(pw50_values) if pw50_values else np.nan,
        "auc_mean": np.mean(auc_values) if auc_values else np.nan,
        "dt_ratio_mean": np.mean(dt_ratios) if dt_ratios else np.nan,
        "perfusion_index": perfusion_index,
        "beats_detected": len(peaks),
        "recording_duration_s": len(ppg) / fs
    }
    return features


# Sanity Check
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


# File handling
def get_latest_processed_file():
    csv_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith("_processed.csv")]
    if not csv_files:
        print("No processed CSV files found.")
        return None
    latest_file = max([os.path.join(PROCESSED_PATH, f) for f in csv_files], key=os.path.getmtime)
    return latest_file


# Main Run Fn
def run_feature_extraction(beats_info, plot=False, save=True):
    latest_file = get_latest_processed_file()
    if latest_file is None:
        return None

    print(f"\n📂 Using latest processed file: {os.path.basename(latest_file)}")
    df = pd.read_csv(latest_file)

    if "ir_filtered" in df.columns:
        ppg = df["ir_filtered"].to_numpy()
    elif "ppg" in df.columns:
        ppg = df["ppg"].to_numpy()
    else:
        print("No valid PPG column found.")
        return None

    fs = 100
    metadata = get_metadata()
    features = extract_ppg_features(ppg, fs, beats_info)

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
        os.makedirs(FEATURES_PATH, exist_ok=True)
        out_path = os.path.join(
            FEATURES_PATH,
            os.path.basename(latest_file).replace("_processed.csv", "_features.csv")
        )
        df_out.to_csv(out_path, index=False)
        print(f"\n💾 Saved extracted features → {out_path}")

    print("\n✅ Extracted Features:")
    print(df_out.T)

    # FIXED: Return the file path so main.py can use it
    return out_path


# MAIN
if __name__ == "__main__":
    print("⚙️ This script expects 'beats_info' to be passed from beat_detection.py")
    print("Import and call run_feature_extraction(beats_info) from your main script.")
