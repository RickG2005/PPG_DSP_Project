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

# Weights
METADATA_WEIGHTS = {
    "family_history_first_degree": 0.45,  # Increased from 0.35
    "age": 0.25,  # Increased from 0.15
    "physical_activity_hrs_per_week": -0.20,  # Increased from -0.10
    "sleep_hours": -0.10,  # Increased from -0.05
    "smoking_status": 0.25  # Increased from 0.12
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
        print(f"  ✓ Family diabetes history detected: +{METADATA_WEIGHTS['family_history_first_degree']:.2f}")

    # Age effect: scaling from 45 upward
    try:
        if age and float(age) >= 45:
            age_factor = METADATA_WEIGHTS["age"] * min(1.0, (float(age) - 45) / 20.0)
            adj += age_factor
            print(f"  ✓ Age risk (≥45): +{age_factor:.2f}")
        elif age and float(age) >= 35:
            # FIXED: Add moderate risk for ages 35-44
            age_factor = METADATA_WEIGHTS["age"] * 0.3 * ((float(age) - 35) / 10.0)
            adj += age_factor
            print(f"  ✓ Age risk (35-44): +{age_factor:.2f}")
    except:
        pass

    # Physical activity: low activity raises risk
    try:
        if activity is not None:
            act = float(activity)
            # FIXED: More aggressive activity penalty
            if act <= 1:  # Sedentary
                activity_penalty = METADATA_WEIGHTS["physical_activity_hrs_per_week"] * -1.0
                adj += activity_penalty
                print(f"  ✓ Sedentary lifestyle: +{activity_penalty:.2f}")
            elif act == 2:  # Moderate
                activity_penalty = METADATA_WEIGHTS["physical_activity_hrs_per_week"] * -0.5
                adj += activity_penalty
                print(f"  ✓ Moderate activity: +{activity_penalty:.2f}")
            else:  # Active
                activity_benefit = METADATA_WEIGHTS["physical_activity_hrs_per_week"] * 0.8
                adj += activity_benefit
                print(f"  ✓ Active lifestyle: {activity_benefit:.2f}")
    except:
        pass

    # Sleep: <6 or >9 hrs increases risk
    try:
        if sleep_h is not None:
            sh = float(sleep_h)
            if sh < 6.0:
                sleep_penalty = 0.08  # Increased from 0.03
                adj += sleep_penalty
                print(f"  ✓ Sleep deprivation (<6h): +{sleep_penalty:.2f}")
            elif 7.0 <= sh <= 9.0:
                sleep_benefit = -0.05  # Increased from -0.02
                adj += sleep_benefit
                print(f"  ✓ Healthy sleep (7-9h): {sleep_benefit:.2f}")
            elif sh > 9.0:
                sleep_penalty = 0.05  # Increased from 0.02
                adj += sleep_penalty
                print(f"  ✓ Excessive sleep (>9h): +{sleep_penalty:.2f}")
    except:
        pass

    # Smoking: strong positive risk
    if str(smoking).lower() in ("1", "yes", "y", "true"):
        adj += METADATA_WEIGHTS["smoking_status"]
        print(f"  ✓ Smoking habit: +{METADATA_WEIGHTS['smoking_status']:.2f}")

    # Wide clamping range 
    final_adj = float(np.clip(adj, -0.5, 0.8))
    print(f"\n  📊 Total Metadata Adjustment: {final_adj:.3f}")
    return final_adj


def physiology_to_absolute_probability(phys_score, metadata_adj, baseline=BASELINE_PREVALENCE):
    """Converts physiology + metadata adjustment → absolute diabetes risk."""
    # Sensitivity multipliers
    k = 4.5  
    phys_mult = 1.0 + k * phys_score
    
    # Metadata adjustment 
    meta_mult = 1.0 + (metadata_adj * 2.0)  # Doubled the impact
    
    combined = phys_mult * meta_mult

    print(f"\n  🧮 Risk Calculation:")
    print(f"     Physiology multiplier: {phys_mult:.3f}")
    print(f"     Metadata multiplier: {meta_mult:.3f}")
    print(f"     Combined multiplier: {combined:.3f}")

    current = float(np.clip(baseline * combined, 0.0, 0.9))
    
    # Projection multipliers
    ten = float(np.clip(current * (1.0 + 0.6 * phys_score + max(0.0, metadata_adj) * 1.0), 0.0, 0.99))
    thirty = float(np.clip(current * (1.0 + 1.5 * phys_score + max(0.0, metadata_adj) * 2.0), 0.0, 0.99))

    return current, ten, thirty


# ---------- MAIN RISK COMPUTATION ----------
def run_diabetes_prediction(physiology_output, save=True):
    features = physiology_output.get("metadata", {})
    phys_score = physiology_output.get("physiology_score", 0.0)

    print("\n🔍 Analyzing metadata risk factors:")
    print("-" * 50)
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
        "risk_label": label,
        "physiology_score": round(phys_score * 100, 2),
        "metadata_adjustment": round(meta_adj * 100, 2)
    }


# ---------- TEST ----------
if __name__ == "__main__":
    # Test with high-risk profile
    sample_data_high_risk = {
        "physiology_score": 0.18,
        "metadata": {
            "family_diabetes": "yes",
            "age": 50,
            "activity_score": 1,  # Sedentary
            "sleep_hours": 5,  # Sleep deprived
            "smoker": "yes"
        }
    }
    
    print("\n" + "="*60)
    print("TEST 1: HIGH RISK PROFILE")
    print("="*60)
    result1 = run_diabetes_prediction(sample_data_high_risk)
    
    # Test with low-risk profile
    sample_data_low_risk = {
        "physiology_score": 0.05,
        "metadata": {
            "family_diabetes": "no",
            "age": 25,
            "activity_score": 3,  # Active
            "sleep_hours": 8,
            "smoker": "no"
        }
    }
    
    print("\n" + "="*60)
    print("TEST 2: LOW RISK PROFILE")
    print("="*60)
    result2 = run_diabetes_prediction(sample_data_low_risk)
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"High Risk Profile: {result1['current_risk']}% current risk")
    print(f"Low Risk Profile: {result2['current_risk']}% current risk")
    print(f"Difference: {result1['current_risk'] - result2['current_risk']:.1f}%")