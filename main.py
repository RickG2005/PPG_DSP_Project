import sys, os
from datetime import datetime
import numpy as np

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_COLLECT_DIR = os.path.join(BASE_DIR, "ppg_data_collect")
SIGNAL_DIR = os.path.join(BASE_DIR, "signal_processing")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")

# Add subfolders to Python’s import path
sys.path.extend([DATA_COLLECT_DIR, SIGNAL_DIR])

# ---------------- IMPORTS ----------------
import preprocess
import beat_detection
import feature_extraction
import feature_normalisation
import risk_model  # ← new module for risk scoring


def get_latest_raw_file():
    """Get the most recent raw CSV from the data/raw folder."""
    if not os.path.exists(RAW_PATH):
        print(f"❌ Raw data folder not found: {RAW_PATH}")
        return None
    csv_files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No raw CSV files found in data/raw.")
        return None
    latest_file = max([os.path.join(RAW_PATH, f) for f in csv_files], key=os.path.getmtime)
    print(f"📂 Using latest raw file: {os.path.basename(latest_file)}")
    return latest_file


def main():
    print("\n🩸 PPG Data Processing Pipeline (ESP32 Bypass Active) 🩸\n")

    # --- Phase 1 (Skipped): Data Collection ---
    print("⚙️ Skipping data collection...")
    raw_file = get_latest_raw_file()
    if not raw_file:
        print("❌ No existing raw file found. Please add one manually.")
        return

    # --- Phase 2: Preprocessing ---
    print("\n[1/5] Preprocessing raw signal...")
    processed_file = preprocess.run_preprocessing(raw_file)
    print(f"✅ Preprocessed data saved → {processed_file}")

    # --- Phase 3: Beat Detection ---
    print("\n[2/5] Detecting beats...")
    beats_data = beat_detection.run_latest_beat_detection(plot=True)
    if beats_data is None:
        print("❌ Beat detection failed.")
        return
    print("✅ Beat detection completed.")

    # --- Phase 4: Feature Extraction ---
    print("\n[3/5] Extracting features...")
    features_file = feature_extraction.run_feature_extraction(beats_data)
    print(f"✅ Feature extraction complete → {features_file}")

            # --- Phase 5A: Feature Normalisation ---
    print("\n[4/5] Normalising features...")
    normalized_features = feature_normalisation.run_feature_normalisation(save=False)
    if normalized_features is None:
        print("❌ Feature normalisation failed.")
        return
    print("✅ Features normalised successfully.")

    # --- Extract physiology score correctly ---
    physiology_score = normalized_features.get("physiology_score", 0.0)

    # --- Example metadata (replace with real input or user questionnaire) ---
    metadata = {
        "family_history_first_degree": "yes",  # or user input
        "age": 26,
        "physical_activity_hrs_per_week": 1,
        "sleep_hours": 6,
        "smoking_status": "yes"
    }

    # --- Prepare input for risk model ---
    physiology_output = {
        "physiology_score": physiology_score,
        "metadata": metadata
    }


    # --- Phase 5B: Risk Assessment ---
    print("\n[5/5] Running diabetes risk assessment...\n")
    result = risk_model.run_diabetes_prediction(physiology_output, save=True)

    # Display result in a nice summary block
    print("\n" + "=" * 50)
    print("🧬 FINAL HEALTH RISK SUMMARY")
    print("=" * 50)
    print(f"🏷️  Risk Category:  {result['risk_label']}")
    print(f"⚡ Current Diabetes Risk: {result['current_risk']}%")
    print(f"⏳ 10-Year Risk: {result['10yr_risk']}%")
    print(f"🧠 30-Year Risk: {result['30yr_risk']}%")
    print("-" * 50)
    print("Breakdown:")
    print(f"  Physiology Score:           {physiology_score:.3f}")
    print(f"  Metadata Adjustments:       {metadata}")
    print("=" * 50)
    print("✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()
