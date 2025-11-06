import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------- CONFIG ----------
FEATURES_PATH = r"C:\Users\rick2\Documents\PPG Project\data\features"
RISK_PATH = r"C:\Users\rick2\Documents\PPG Project\data\risk"

# ---------- HELPERS ----------
def safe_get(d, k):
    v = d.get(k)
    return None if (pd.isna(v) or v is None) else v

def risk_from_deviation(value, normal_min, normal_max, weight=1.0):
    if value is None:
        return 0.0
    try:
        val = float(value)
    except:
        return 0.0
    if normal_min <= val <= normal_max:
        return 0.0
    dist = min(abs(val - normal_min), abs(val - normal_max))
    norm_range = max(1e-6, (normal_max - normal_min))
    score = (dist / norm_range) * weight
    return float(np.clip(score, 0.0, 1.0))

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

# ---------- WEIGHTS & RANGES ----------
PPG_WEIGHTS = {
    "heart_rate_bpm": 0.22,
    "hrv_ms": 0.25,
    "amplitude_mean": 0.18,
    "rise_time_mean_s": 0.18,
    "bmi": 0.17
}

HEALTH_RANGES = {
    "heart_rate_bpm": (60, 100),
    "hrv_ms": (20, 120),
    "amplitude_mean": (0.2, 1.2),
    "rise_time_mean_s": (0.12, 0.6),
    "bmi": (18.5, 24.9)
}

# ---------- CORE ----------
def compute_physiology_score(features):
    total_w = 0.0
    score = 0.0
    breakdown = {}
    for f, w in PPG_WEIGHTS.items():
        lo_hi = HEALTH_RANGES.get(f)
        val = safe_get(features, f)
        if lo_hi:
            lo, hi = lo_hi
            r = risk_from_deviation(val, lo, hi, weight=1.0)
        else:
            r = 0.0
        breakdown[f] = round(r * 100, 2)
        score += r * w
        total_w += w
    if total_w <= 0:
        return 0.0, breakdown
    phys_score = clamp01(score / total_w)
    return phys_score, breakdown

def get_latest_feature_file():
    csv_files = [f for f in os.listdir(FEATURES_PATH) if f.endswith("_features.csv")]
    if not csv_files:
        print("❌ No feature files found.")
        return None
    return max([os.path.join(FEATURES_PATH, f) for f in csv_files], key=os.path.getmtime)

def run_feature_normalisation(save=True):
    latest = get_latest_feature_file()
    if latest is None:
        return None

    print(f"\n📂 Using latest features file: {os.path.basename(latest)}")
    df = pd.read_csv(latest)
    features = df.iloc[0].to_dict()

    phys_score, phys_breakdown = compute_physiology_score(features)

    # FIXED: Extract metadata from features file
    metadata = {
        "age": features.get("age"),
        "sex": features.get("sex"),
        "weight_kg": features.get("weight_kg"),
        "height_cm": features.get("height_cm"),
        "bmi": features.get("bmi"),
        "sleep_hours": features.get("sleep_hours"),
        "activity_level": features.get("activity_level"),
        "activity_score": features.get("activity_score"),
        "family_diabetes": features.get("family_diabetes"),
        "smoker": features.get("smoker"),
        "caffeine_intake": features.get("caffeine_intake")
    }

    # Save the normalized physiology summary
    if save:
        os.makedirs(RISK_PATH, exist_ok=True)
        base = os.path.basename(latest).replace("_features.csv", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_risk = os.path.join(RISK_PATH, f"{base}_physiology_{timestamp}.csv")
        pd.DataFrame([
            {"Metric": "Physiology Score (%)", "Value": round(phys_score * 100, 2)},
            *[{"Metric": f"Component: {k}", "Value": v} for k, v in phys_breakdown.items()]
        ]).to_csv(out_risk, index=False)
        print(f"💾 Saved normalized physiology → {out_risk}")

    # FIXED: Return both physiology score and metadata
    return {
        "physiology_score": round(phys_score, 4),
        "breakdown": phys_breakdown,
        "metadata": metadata,  # Now includes actual user metadata
        "source_features": latest
    }

if __name__ == "__main__":
    result = run_feature_normalisation(save=True)
    if result:
        print(f"\nPhysiology Score: {result['physiology_score']}")
        print(f"Metadata: {result['metadata']}")