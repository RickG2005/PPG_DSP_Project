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
# These represent *max* per-item contributions (not direct % points).
# Family and smoking are strongest; activity and sleep are modifiable protective/risk factors.
METADATA_WEIGHTS = {
    "family_history_first_degree": 0.45,   # strong hereditary effect
    "age": 0.25,                           # scaled by age bracket
    "physical_activity_hrs_per_week": -0.20,  # protective when active, penalty when sedentary
    "sleep_hours": -0.10,                  # healthy sleep reduces risk; extremes increase it
    "smoking_status": 0.35                 # substantial increased risk for smokers
}

console = Console()


# ---------- CORE FUNCTIONS ----------
def compute_metadata_adjustment(features):
    """
    Compute a metadata adjustment value where:
      - negative values lower risk (protective)
      - positive values raise risk
    This function makes each metadata entry affect risk differently,
    using medically reasonable brackets/coefficients.

    Returns final_adj (float) already clamped to a safe range.
    """

    # Accept both old/new keys
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
    print("\n  📌 Metadata detail:")

    # ----- Family history (binary) -----
    try:
        if str(family).lower() in ("1", "yes", "y", "true"):
            adj += METADATA_WEIGHTS["family_history_first_degree"]
            print(f"  ✓ Family diabetes history: +{METADATA_WEIGHTS['family_history_first_degree']:.2f}")
    except:
        pass

    # ----- Age: graded effect -----
    try:
        if age is not None:
            a = float(age)
            if a >= 70:
                age_add = METADATA_WEIGHTS["age"] * 1.0        # full weight
                adj += age_add
                print(f"  ✓ Age ≥70: +{age_add:.2f}")
            elif a >= 60:
                age_add = METADATA_WEIGHTS["age"] * 0.85
                adj += age_add
                print(f"  ✓ Age 60–69: +{age_add:.2f}")
            elif a >= 50:
                age_add = METADATA_WEIGHTS["age"] * 0.6
                adj += age_add
                print(f"  ✓ Age 50–59: +{age_add:.2f}")
            elif a >= 45:
                age_add = METADATA_WEIGHTS["age"] * 0.35
                adj += age_add
                print(f"  ✓ Age 45–49: +{age_add:.2f}")
            elif a >= 35:
                age_add = METADATA_WEIGHTS["age"] * 0.10
                adj += age_add
                print(f"  ✓ Age 35–44: +{age_add:.2f}")
    except:
        pass

    # ----- Physical activity: hours per week or score -----
    # Interpret numerical meaning:
    #  <=1 : sedentary (high penalty)
    #   2  : low-moderate
    #  3-4 : moderate
    #  >=4 : active (protective)
    try:
        if activity is not None:
            act = float(activity)
            if act <= 1:
                act_adj = abs(METADATA_WEIGHTS["physical_activity_hrs_per_week"]) * 1.0
                adj += act_adj
                print(f"  ✓ Sedentary (≤1h/wk): +{act_adj:.2f}")
            elif act == 2:
                act_adj = abs(METADATA_WEIGHTS["physical_activity_hrs_per_week"]) * 0.5
                adj += act_adj
                print(f"  ✓ Low activity (≈2h/wk): +{act_adj:.2f}")
            elif 3 <= act < 4:
                act_adj = METADATA_WEIGHTS["physical_activity_hrs_per_week"] * 0.3
                adj += act_adj
                print(f"  ✓ Moderate activity (3–4h/wk): {act_adj:.2f}")
            else:  # act >= 4
                act_adj = METADATA_WEIGHTS["physical_activity_hrs_per_week"] * 0.8
                adj += act_adj
                print(f"  ✓ Active (≥4h/wk): {act_adj:.2f}")
    except:
        pass

    # ----- Sleep: graded and asymmetric -----
    # <4h severe penalty, 4-6h penalty, 7-9h protective, >9h mild penalty
    try:
        if sleep_h is not None:
            sh = float(sleep_h)
            if sh < 4.0:
                s_adj = 0.10
                adj += s_adj
                print(f"  ✓ Severe sleep loss (<4h): +{s_adj:.2f}")
            elif sh < 6.0:
                s_adj = 0.06
                adj += s_adj
                print(f"  ✓ Short sleep (4–6h): +{s_adj:.2f}")
            elif 7.0 <= sh <= 9.0:
                s_adj = METADATA_WEIGHTS["sleep_hours"] * 0.8  # protective
                adj += s_adj
                print(f"  ✓ Healthy sleep (7–9h): {s_adj:.2f}")
            elif sh > 9.0:
                s_adj = 0.05
                adj += s_adj
                print(f"  ✓ Long sleep (>9h): +{s_adj:.2f}")
    except:
        pass

    # ----- Smoking (binary) -----
    try:
        if str(smoking).lower() in ("1", "yes", "y", "true"):
            adj += METADATA_WEIGHTS["smoking_status"]
            print(f"  ✓ Smoking habit: +{METADATA_WEIGHTS['smoking_status']:.2f}")
    except:
        pass

    # ----- Final clamp: ensure metadata adj doesn't produce wild swings -----
    # We allow metadata to move risk substantially, but keep within safe numeric bounds.
    # Negative = protective; Positive = increased risk.
    final_adj = float(np.clip(adj, -0.35, 0.90))
    print(f"\n  📊 Total Metadata Adjustment (clamped): {final_adj:.3f}")
    return final_adj


def physiology_to_absolute_probability(phys_score, metadata_adj, baseline=BASELINE_PREVALENCE):
    """
    Convert physiology + metadata adjustment to absolute current/10yr/30yr risk.
    Design choices:
      - Physiology influences ~60% of the modulating factor.
      - Metadata influences ~40% of the modulating factor.
      - Use a combined linear mixture inside a multiplier applied to baseline.
      - Current risk hard-capped at 50% to match your request.
    """

    # Defensive cap on phys_score (keeps sensor noise from exploding risk)
    phys_score = float(max(0.0, min(phys_score, 1.0)))

    # Scaling constants (tuned)
    phys_scale = 1.8   # how a unit phys_score moves the risk when weighted
    meta_scale = 1.2   # how a unit metadata_adj moves the risk when weighted

    # Weighted combination (physiology dominant ~60%)
    weighted_phys = 0.60 * (phys_score * phys_scale)
    weighted_meta = 0.40 * (metadata_adj * meta_scale)

    combined_factor = 1.0 + weighted_phys + weighted_meta

    print(f"\n  🧮 Risk Calculation:")
    print(f"     Physiology component (0.60 * scale): {weighted_phys:.3f}")
    print(f"     Metadata component  (0.40 * scale): {weighted_meta:.3f}")
    print(f"     Combined multiplier: {combined_factor:.3f}")

    # Current risk (hard cap at 50% as requested)
    current = float(np.clip(baseline * combined_factor, 0.0, 0.50))

    # Projected risks: applied conservatively with caps
    ten = float(np.clip(current * (1.0 + 0.40 * phys_score + max(0.0, metadata_adj) * 0.6), 0.0, 0.65))
    thirty = float(np.clip(current * (1.0 + 0.85 * phys_score + max(0.0, metadata_adj) * 1.1), 0.0, 0.80))

    return current, ten, thirty


# ---------- MAIN RISK COMPUTATION ----------
def run_diabetes_prediction(physiology_output, save=True):
    features = physiology_output.get("metadata", {})
    phys_score = physiology_output.get("physiology_score", 0.0)

    # CAP physiology score if it's unrealistically high due to sensor issues
    if phys_score > 0.6:
        print(f"\n⚠️  Warning: Physiology score very high ({phys_score*100:.1f}%)")
        print(f"   This may be due to sensor signal quality issues.")
        print(f"   Capping at 60% for realistic risk assessment.\n")
        phys_score = 0.6

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
    # Test with WORST CASE profile (should be max ~50% current and ≤65% 10-year)
    sample_data_worst = {
        "physiology_score": 0.50,  # High but realistic
        "metadata": {
            "family_diabetes": "yes",
            "age": 65,
            "activity_score": 1,  # Sedentary
            "sleep_hours": 3,     # Severe deprivation
            "smoker": "yes"
        }
    }

    print("\n" + "="*60)
    print("TEST 1: WORST CASE PROFILE (Should be ≤50% current, ≤65% 10-year)")
    print("="*60)
    result1 = run_diabetes_prediction(sample_data_worst)

    # Test with high-risk profile
    sample_data_high_risk = {
        "physiology_score": 0.30,
        "metadata": {
            "family_diabetes": "yes",
            "age": 50,
            "activity_score": 1,  # Sedentary
            "sleep_hours": 5,
            "smoker": "yes"
        }
    }

    print("\n" + "="*60)
    print("TEST 2: HIGH RISK PROFILE")
    print("="*60)
    result2 = run_diabetes_prediction(sample_data_high_risk)

    # Test with low-risk profile
    sample_data_low_risk = {
        "physiology_score": 0.05,
        "metadata": {
            "family_diabetes": "no",
            "age": 25,
            "activity_score": 4,  # Active
            "sleep_hours": 8,
            "smoker": "no"
        }
    }

    print("\n" + "="*60)
    print("TEST 3: LOW RISK PROFILE")
    print("="*60)
    result3 = run_diabetes_prediction(sample_data_low_risk)

    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"Worst Case:   {result1['10yr_risk']}% (10-year) - Should be ≤65%")
    print(f"High Risk:    {result2['10yr_risk']}% (10-year)")
    print(f"Low Risk:     {result3['10yr_risk']}% (10-year)")
    print(f"\nRange: {result3['10yr_risk']:.1f}% to {result1['10yr_risk']:.1f}%")
