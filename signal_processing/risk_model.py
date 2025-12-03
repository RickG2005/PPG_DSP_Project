import os
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table

# ---------- CONFIG ----------
RISK_PATH = r"C:\Users\rick2\Documents\PPG Project\data\risk"
FINAL_PATH = r"C:\Users\rick2\Documents\PPG Project\data\final"

BASELINE_PREVALENCE = 0.12  # global baseline prevalence (12%)

# ---------- MEDICALLY-ASSESSED METADATA WEIGHTS ----------
METADATA_WEIGHTS = {
    "family_history_first_degree": 0.45,
    "age": 0.25,
    "physical_activity_hrs_per_week": -0.20,
    "sleep_hours": -0.10,
    "smoking_status": 0.35
}

console = Console()

# ---------- CORE FUNCTIONS ----------
def compute_metadata_adjustment(features):
    family = features.get("family_history_first_degree") or features.get("family_diabetes") or features.get("family_history")
    smoking = features.get("smoking_status") or features.get("smoker")
    age = features.get("age")
    sleep_h = features.get("sleep_hours")
    activity = features.get("physical_activity_hrs_per_week") or features.get("activity_score")

    adj = 0.0
    print("\n  📌 Metadata detail:")

    # Family history
    if str(family).lower() in ("1", "yes", "y", "true"):
        adj += METADATA_WEIGHTS["family_history_first_degree"]
        print(f"  ✓ Family diabetes history: +{METADATA_WEIGHTS['family_history_first_degree']:.2f}")

    # Age
    if age is not None:
        a = float(age)
        if a >= 70:
            adj += METADATA_WEIGHTS["age"]
            print(f"  ✓ Age ≥70: +{METADATA_WEIGHTS['age']:.2f}")
        elif a >= 60:
            adj += METADATA_WEIGHTS["age"] * 0.85
            print(f"  ✓ Age 60–69: +{METADATA_WEIGHTS['age']*0.85:.2f}")
        elif a >= 50:
            adj += METADATA_WEIGHTS["age"] * 0.6
            print(f"  ✓ Age 50–59: +{METADATA_WEIGHTS['age']*0.6:.2f}")
        elif a >= 45:
            adj += METADATA_WEIGHTS["age"] * 0.35
            print(f"  ✓ Age 45–49: +{METADATA_WEIGHTS['age']*0.35:.2f}")
        elif a >= 35:
            adj += METADATA_WEIGHTS["age"] * 0.1
            print(f"  ✓ Age 35–44: +{METADATA_WEIGHTS['age']*0.1:.2f}")

    # Physical activity
    if activity is not None:
        act = float(activity)
        if act <= 1:
            adj += abs(METADATA_WEIGHTS["physical_activity_hrs_per_week"])
            print(f"  ✓ Sedentary (≤1h/wk): +{abs(METADATA_WEIGHTS['physical_activity_hrs_per_week']):.2f}")
        elif act == 2:
            adj += abs(METADATA_WEIGHTS["physical_activity_hrs_per_week"]) * 0.5
            print(f"  ✓ Low activity (≈2h/wk): +{abs(METADATA_WEIGHTS['physical_activity_hrs_per_week'])*0.5:.2f}")
        elif 3 <= act < 4:
            adj += METADATA_WEIGHTS["physical_activity_hrs_per_week"] * 0.3
            print(f"  ✓ Moderate activity (3–4h/wk): {METADATA_WEIGHTS['physical_activity_hrs_per_week']*0.3:.2f}")
        else:
            adj += METADATA_WEIGHTS["physical_activity_hrs_per_week"] * 0.8
            print(f"  ✓ Active (≥4h/wk): {METADATA_WEIGHTS['physical_activity_hrs_per_week']*0.8:.2f}")

    # Sleep
    if sleep_h is not None:
        sh = float(sleep_h)
        if sh < 4.0:
            adj += 0.10
            print(f"  ✓ Severe sleep loss (<4h): +0.10")
        elif sh < 6.0:
            adj += 0.06
            print(f"  ✓ Short sleep (4–6h): +0.06")
        elif 7.0 <= sh <= 9.0:
            adj += METADATA_WEIGHTS["sleep_hours"] * 0.8
            print(f"  ✓ Healthy sleep (7–9h): {METADATA_WEIGHTS['sleep_hours']*0.8:.2f}")
        elif sh > 9.0:
            adj += 0.05
            print(f"  ✓ Long sleep (>9h): +0.05")

    # Smoking
    if str(smoking).lower() in ("1", "yes", "y", "true"):
        adj += METADATA_WEIGHTS["smoking_status"]
        print(f"  ✓ Smoking habit: +{METADATA_WEIGHTS['smoking_status']:.2f}")

    final_adj = float(np.clip(adj, -0.35, 0.90))
    print(f"\n  📊 Total Metadata Adjustment (clamped): {final_adj:.3f}")
    return final_adj

def physiology_to_absolute_probability(phys_score, metadata_adj, baseline=BASELINE_PREVALENCE):
    phys_score = float(max(0.0, min(phys_score, 1.0)))
    phys_scale = 1.8
    meta_scale = 1.2

    weighted_phys = 0.60 * (phys_score * phys_scale)
    weighted_meta = 0.40 * (metadata_adj * meta_scale)
    combined_factor = 1.0 + weighted_phys + weighted_meta

    current = float(np.clip(baseline * combined_factor, 0.0, 0.50))
    ten = float(np.clip(current * (1.0 + 0.40 * phys_score + max(0.0, metadata_adj) * 0.6), 0.0, 0.65))
    thirty = float(np.clip(current * (1.0 + 0.85 * phys_score + max(0.0, metadata_adj) * 1.1), 0.0, 0.80))

    return current, ten, thirty

def run_diabetes_prediction(physiology_output, save=True):
    features = physiology_output.get("metadata", {})
    phys_score = physiology_output.get("physiology_score", 0.0)

    if phys_score > 0.6:
        phys_score = 0.6

    meta_adj = compute_metadata_adjustment(features)
    current, p10, p30 = physiology_to_absolute_probability(phys_score, meta_adj, BASELINE_PREVALENCE)

    table = Table(title="🩸 Diabetes Risk Analysis Report", style="bold cyan")
    table.add_column("Metric", justify="left", style="bold white")
    table.add_column("Value", justify="right", style="bold yellow")
    table.add_row("Current Likelihood", f"{current*100:.1f}%")
    table.add_row("10-Year Projected Risk", f"{p10*100:.1f}%")
    table.add_row("30-Year Projected Risk", f"{p30*100:.1f}%")
    table.add_row("Physiology Score", f"{phys_score*100:.1f}%")
    table.add_row("Metadata Adjustment", f"{meta_adj*100:.1f}%")

    console.print("\n")
    console.print(table)
    console.print("\n")

    if save:
        os.makedirs(FINAL_PATH, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_final = os.path.join(FINAL_PATH, f"final_diabetes_risk_{timestamp}.csv")

        # Save metrics + metadata
        output_data = [
            {"Metric": "Current Likelihood (%)", "Value": round(current * 100, 2)},
            {"Metric": "10-Year Risk (%)", "Value": round(p10 * 100, 2)},
            {"Metric": "30-Year Risk (%)", "Value": round(p30 * 100, 2)},
            {"Metric": "Physiology Score (%)", "Value": round(phys_score * 100, 2)},
            {"Metric": "Metadata Adjustment (%)", "Value": round(meta_adj * 100, 2)}
        ]
        for k, v in features.items():
            output_data.append({"Metric": k, "Value": v})

        pd.DataFrame(output_data).to_csv(out_final, index=False)
        console.print(f"💾 [green]Saved final report → {out_final}[/green]\n")

    label = (
        "Very Low Risk" if current * 100 < 5 else
        "Low Risk" if current * 100 < 15 else
        "Moderate Risk" if current * 100 < 30 else
        "High Risk"
    )

    return {
        "current_risk": round(current * 100, 2),
        "10yr_risk": round(p10 * 100, 2),
        "30yr_risk": round(p30 * 100, 2),
        "risk_label": label,
        "physiology_score": round(phys_score * 100, 2),
        "metadata_adjustment": round(meta_adj * 100, 2),
        "metadata": features
    }
