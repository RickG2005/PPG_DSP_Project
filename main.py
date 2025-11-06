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
import preprocess
import beat_detection
import feature_extraction
import feature_normalisation
import risk_model


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
    print("\n" + "="*60)
    print("🩸 PPG DIABETES RISK ASSESSMENT PIPELINE")
    print("="*60 + "\n")

    # --- Phase 1 (Skipped): Data Collection ---
    print("⚙️ Skipping data collection...")
    raw_file = get_latest_raw_file()
    if not raw_file:
        print("❌ No existing raw file found. Please add one manually.")
        return

    # --- Phase 2: Preprocessing ---
    print("\n" + "-"*60)
    print("[1/5] PREPROCESSING RAW SIGNAL")
    print("-"*60)
    processed_file = preprocess.run_preprocessing(raw_file)
    print(f"✅ Preprocessed data saved → {processed_file}")

    # --- Phase 3: Beat Detection ---
    print("\n" + "-"*60)
    print("[2/5] DETECTING HEARTBEATS")
    print("-"*60)
    beats_data = beat_detection.run_latest_beat_detection(plot=True)
    if beats_data is None:
        print("❌ Beat detection failed.")
        return
    print("✅ Beat detection completed.")

    # --- Phase 4: Feature Extraction (with metadata input) ---
    print("\n" + "-"*60)
    print("[3/5] EXTRACTING FEATURES & COLLECTING METADATA")
    print("-"*60)
    features_file = feature_extraction.run_feature_extraction(beats_data, save=True)
    if features_file is None:
        print("❌ Feature extraction failed.")
        return
    print(f"✅ Feature extraction complete → {features_file}")

    # --- Phase 5A: Feature Normalisation ---
    print("\n" + "-"*60)
    print("[4/5] NORMALISING PHYSIOLOGICAL FEATURES")
    print("-"*60)
    normalized_output = feature_normalisation.run_feature_normalisation(save=True)
    if normalized_output is None:
        print("❌ Feature normalisation failed.")
        return
    
    physiology_score = normalized_output.get("physiology_score", 0.0)
    metadata = normalized_output.get("metadata", {})
    
    print("✅ Features normalised successfully.")
    print(f"   Physiology Score: {physiology_score:.4f}")

    # --- Phase 5B: Risk Assessment ---
    print("\n" + "-"*60)
    print("[5/5] COMPUTING DIABETES RISK")
    print("-"*60)
    
    # Use metadata
    physiology_output = {
        "physiology_score": physiology_score,
        "metadata": metadata  # This now contains the real user input
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