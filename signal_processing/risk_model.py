import os
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table

# ---------- CONFIG ----------
RISK_PATH = r"C:\Users\rick2\Documents\PPG Project\data\risk"
FINAL_PATH = r"C:\Users\rick2\Documents\PPG Project\data\final"

BASELINE_PREVALENCE = 0.12  # global baseline prevalence

METADATA_WEIGHTS = {
    "family_history_first_degree": 0.35,
    "age": 0.15,
    "physical_activity_hrs_per_week": -0.10,
    "sleep_hours": -0.05,
    "smoking_status": 0.12
}

console = Console()


# ---------- CORE FUNCTIONS ----------
def compute_metadata_adjustment(features):
    """Computes adjustment factor using both old and new key mappings."""
    # Support both metadata naming styles
    family = (
        features.get("family_history_first_degree")
        or features.get("family_diabetes")
        or features.get("family_history")
    )

    smoking = (
        features.get("smoking_status")
        or features.get("smoker")
    )

    age = features.get("age")
    sleep_h = features.get("sleep_hours")
    activity = (
        features.get("physical_activity_hrs_per_week")
        or features.get("activity_score")
    )

    adj = 0.0

    # Family history: strong positive risk
    if str(family).lower() in ("1", "yes", "y", "true"):
        adj += METADATA_WEIGHTS["family_history_first_degree"]

    # Age effect: scaling from 45 upward
    try:
        if age and float(age) >= 45:
            adj += METADATA_WEIGHTS["age"] * min(1.0, (float(age) - 45) / 20.0)
    except:
        pass

    # Physical activity: low activity raises risk
    try:
        if activity is not None:
            act = float(activity)
            adj += METADATA_WEIGHTS["physical_activity_hrs_per_week"] * (1.0 - min(1.0, act / 5.0))
    except:
        pass

    # Sleep: <6 or >9 hrs slightly increases risk
    try:
        if sleep_h is not None:
            sh = float(sleep_h)
            if sh < 6.0:
                adj += 0.03
            elif 7.0 <= sh <= 9.0:
                adj -= 0.02
            elif sh > 9.0:
                adj += 0.02
    except:
        pass

    # Smoking: moderate positive risk
    if str(smoking).lower() in ("1", "yes", "y", "true"):
        adj += METADATA_WEIGHTS["smoking_status"]

    return float(np.clip(adj, -0.3, 0.6))


def physiology_to_absolute_probability(phys_score, metadata_adj, baseline=BASELINE_PREVALENCE):
    """Converts physiology + metadata adjustment → absolute diabetes risk."""
    k = 2.7
    phys_mult = 1.0 + k * phys_score
    meta_mult = 1.0 + metadata_adj
    combined = phys_mult * meta_mult

    current = float(np.clip(baseline * combined, 0.0, 0.9))
    ten = float(np.clip(current * (1.0 + 0.4 * phys_score + max(0.0, metadata_adj) * 0.6), 0.0, 0.99))
    thirty = float(np.clip(current * (1.0 + 1.1 * phys_score + max(0.0, metadata_adj) * 1.4), 0.0, 0.99))

    return current, ten, thirty


# ---------- MAIN RISK COMPUTATION ----------
def run_diabetes_prediction(physiology_output, save=True):
    features = physiology_output.get("metadata", {})
    phys_score = physiology_output.get("physiology_score", 0.0)

    meta_adj = compute_metadata_adjustment(features)
    current, p10, p30 = physiology_to_absolute_probability(phys_score, meta_adj, BASELINE_PREVALENCE)

    # -------- Display results --------
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

    # -------- Save results --------
    if save:
        os.makedirs(FINAL_PATH, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_final = os.path.join(FINAL_PATH, f"final_diabetes_risk_{timestamp}.csv")
        pd.DataFrame([
            {"Metric": "Current Likelihood (%)", "Value": round(current * 100, 2)},
            {"Metric": "10-Year Risk (%)", "Value": round(p10 * 100, 2)},
            {"Metric": "30-Year Risk (%)", "Value": round(p30 * 100, 2)},
            {"Metric": "Physiology Score (%)", "Value": round(phys_score * 100, 2)},
            {"Metric": "Metadata Adjustment (%)", "Value": round(meta_adj * 100, 2)},
        ]).to_csv(out_final, index=False)
        console.print(f"💾 [green]Saved final report → {out_final}[/green]\n")

    # -------- Label --------
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
        "risk_label": label
    }


# ---------- TEST ----------
if __name__ == "__main__":
    sample_data = {
        "physiology_score": 0.18,
        "metadata": {
            "family_diabetes": "yes",
            "age": 26,
            "activity_score": 1,
            "sleep_hours": 6,
            "smoker": "yes"
        }
    }
    run_diabetes_prediction(sample_data)
