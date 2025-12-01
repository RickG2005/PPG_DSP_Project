import sys, os
from datetime import datetime
import numpy as np

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_COLLECT_DIR = os.path.join(BASE_DIR, "ppg_data_collect")
SIGNAL_DIR = os.path.join(BASE_DIR, "signal_processing")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")

# Add subfolders to Python's import path
sys.path.extend([DATA_COLLECT_DIR, SIGNAL_DIR])

# ---------------- IMPORTS ----------------
import data_to_csv
import preprocess
import beat_detection
import feature_extraction
import feature_normalisation
import risk_model


def main():
    print("\n" + "="*60)
    print("🩸 PPG DIABETES RISK ASSESSMENT PIPELINE")
    print("="*60 + "\n")

    # --- Phase 1: Data Collection ---
    print("-"*60)
    print("[1/6] COLLECTING PPG DATA FROM ESP32 SENSOR")
    print("-"*60)
    raw_file = data_to_csv.run_data_collection()
    if not raw_file:
        print("❌ Data collection failed. No raw file generated.")
        return
    print(f"✅ Data collection complete → {raw_file}")

    # --- Phase 2: Preprocessing ---
    print("\n" + "-"*60)
    print("[2/6] PREPROCESSING RAW SIGNAL")
    print("-"*60)
    processed_file = preprocess.run_preprocessing(raw_file)
    print(f"✅ Preprocessed data saved → {processed_file}")

    # --- Phase 3: Beat Detection ---
    print("\n" + "-"*60)
    print("[3/6] DETECTING HEARTBEATS")
    print("-"*60)
    beats_data = beat_detection.run_latest_beat_detection(plot=True)
    if beats_data is None:
        print("❌ Beat detection failed.")
        return
    print("✅ Beat detection completed.")

    # --- Phase 4: Feature Extraction (with metadata input) ---
    print("\n" + "-"*60)
    print("[4/6] EXTRACTING FEATURES & COLLECTING METADATA")
    print("-"*60)
    features_file = feature_extraction.run_feature_extraction(beats_data, save=True)
    if features_file is None:
        print("❌ Feature extraction failed.")
        return
    print(f"✅ Feature extraction complete → {features_file}")

    # --- Phase 5: Feature Normalisation ---
    print("\n" + "-"*60)
    print("[5/6] NORMALISING PHYSIOLOGICAL FEATURES")
    print("-"*60)
    normalized_output = feature_normalisation.run_feature_normalisation(save=True)
    if normalized_output is None:
        print("❌ Feature normalisation failed.")
        return
    
    physiology_score = normalized_output.get("physiology_score", 0.0)
    metadata = normalized_output.get("metadata", {})
    
    print("✅ Features normalised successfully.")
    print(f"   Physiology Score: {physiology_score:.4f}")

    # --- Phase 6: Risk Assessment ---
    print("\n" + "-"*60)
    print("[6/6] COMPUTING DIABETES RISK")
    print("-"*60)
    
    # Use metadata from feature extraction
    physiology_output = {
        "physiology_score": physiology_score,
        "metadata": metadata
    }

    result = risk_model.run_diabetes_prediction(physiology_output, save=True)

    # Display final summary
    print("\n" + "="*60)
    print("🧬 FINAL HEALTH RISK SUMMARY")
    print("="*60)
    print(f"🏷️  Risk Category:           {result['risk_label']}")
    print(f"⚡ Current Diabetes Risk:    {result['current_risk']}%")
    print(f"⏳ 10-Year Projected Risk:   {result['10yr_risk']}%")
    print(f"🧠 30-Year Projected Risk:   {result['30yr_risk']}%")
    print("-" * 60)
    print("📊 Breakdown:")
    print(f"   Physiology Score:         {result['physiology_score']:.2f}%")
    print(f"   Metadata Adjustment:      {result['metadata_adjustment']:.2f}%")
    print("-" * 60)
    print("👤 User Profile:")
    for key, value in metadata.items():
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            print(f"   {key:25s}: {value}")
    print("="*60)
    print("✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()