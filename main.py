import sys, os
from datetime import datetime
import numpy as np
from flask import Flask, render_template
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
DISPLAY_DIR = os.path.join(BASE_DIR, "display")
TEMPLATES_DIR = os.path.join(DISPLAY_DIR, "templates")
DATA_COLLECT_DIR = os.path.join(BASE_DIR, "ppg_data_collect")
SIGNAL_DIR = os.path.join(BASE_DIR, "signal_processing")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
FINAL_PATH = os.path.join(BASE_DIR, "data", "final")

sys.path.extend([DATA_COLLECT_DIR, SIGNAL_DIR])

import data_to_csv
import preprocess
import beat_detection
import feature_extraction
import feature_normalisation
import risk_model

#Flask
app = Flask(__name__, template_folder=TEMPLATES_DIR)

def get_latest_final_csv():
    csv_files = [f for f in os.listdir(FINAL_PATH) if f.endswith(".csv")]
    if not csv_files:
        return None
    latest_file = max([os.path.join(FINAL_PATH, f) for f in csv_files], key=os.path.getmtime)
    return latest_file

@app.route("/")
def index():
    latest_file = get_latest_final_csv()
    if not latest_file:
        return "<h2>No final data found.</h2>"

    df = pd.read_csv(latest_file)
    records = df.to_dict(orient="records")

    # Separate risk metrics from metadata
    risk_keys = {
        "Current Likelihood (%)": "current_risk",
        "10-Year Risk (%)": "10yr_risk",
        "30-Year Risk (%)": "30yr_risk",
        "Physiology Score (%)": "physiology_score",
        "Metadata Adjustment (%)": "metadata_adjustment"
    }

    risk = {}
    metadata = {}
    for row in records:
        key = row["Metric"]
        value = row["Value"]
        if key in risk_keys:
            risk[risk_keys[key]] = value
        else:
            metadata[key] = value

    # Assign risk label
    current = float(risk.get("current_risk", 0))
    if current < 5:
        risk["risk_label"] = "Very Low Risk"
    elif current < 15:
        risk["risk_label"] = "Low Risk"
    elif current < 30:
        risk["risk_label"] = "Moderate Risk"
    else:
        risk["risk_label"] = "High Risk"

    return render_template(
        "index.html",
        filename=os.path.basename(latest_file),
        risk=risk,
        metadata=metadata
    )

# Pipeline
def run_pipeline():
    print("\n" + "="*60)
    print("🩸 PPG DIABETES RISK ASSESSMENT PIPELINE")
    print("="*60 + "\n")

    # Data Collection
    print("-"*60)
    print("[1/6] COLLECTING PPG DATA FROM ESP32 SENSOR")
    print("-"*60)
    raw_file = data_to_csv.run_data_collection()
    if not raw_file:
        print("❌ Data collection failed. No raw file generated.")
        return
    print(f"✅ Data collection complete → {raw_file}")

    # Preprocessing 
    print("\n" + "-"*60)
    print("[2/6] PREPROCESSING RAW SIGNAL")
    print("-"*60)
    processed_file = preprocess.run_preprocessing(raw_file)
    print(f"✅ Preprocessed data saved → {processed_file}")

    # Beat Detection 
    print("\n" + "-"*60)
    print("[3/6] DETECTING HEARTBEATS")
    print("-"*60)
    beats_data = beat_detection.run_latest_beat_detection(plot=True)
    if beats_data is None:
        print("❌ Beat detection failed.")
        return
    print("✅ Beat detection completed.")

    # Phase 4: Feature Extraction 
    print("\n" + "-"*60)
    print("[4/6] EXTRACTING FEATURES & COLLECTING METADATA")
    print("-"*60)
    features_file = feature_extraction.run_feature_extraction(beats_data, save=True)
    if features_file is None:
        print("❌ Feature extraction failed.")
        return
    print(f"✅ Feature extraction complete → {features_file}")

    # Phase 5: Feature Normalisation 
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

    # Phase 6: Risk Assessment 
    print("\n" + "-"*60)
    print("[6/6] COMPUTING DIABETES RISK")
    print("-"*60)
    
    physiology_output = {
        "physiology_score": physiology_score,
        "metadata": metadata
    }

    result = risk_model.run_diabetes_prediction(physiology_output, save=True)

    # Display CLI summary
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
    # Run the CLI pipeline
    run_pipeline()

    # Ask user if they want to see the web page
    choice = input("\n🌐 Open web page with latest report? (y/n): ").strip().lower()
    if choice in ("y", "yes"):
        print("🚀 Starting Flask web server at http://127.0.0.1:5000 ...")
        app.run(debug=False)
    else:
        print("❌ Web server not started. Pipeline finished.")
