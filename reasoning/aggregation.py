import yaml
import os

# Load Config
config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    AGG_CONFIG = config.get('aggregation', {})
except:
    AGG_CONFIG = {}

def aggregate_decision(claim_obj):
    """
    Applies the 4-Stage Deterministic Gate Logic.
    Returns: (prediction_int, rationale_str)
    """
    verdict = claim_obj.judge_verdict
    status = verdict.get("status", "Uncertain").lower()
    
    # Safely parse confidence
    try:
        confidence = float(verdict.get("confidence", 0.0))
    except:
        confidence = 0.0
        
    risk = claim_obj.risk_tier
    key_point = verdict.get("key_point", "No rationale")

    # Load Tuned Thresholds from Config
    TH_CONSISTENCY = AGG_CONFIG.get('consistency_threshold', 0.55)
    TH_UNCERTAINTY = AGG_CONFIG.get('uncertainty_threshold', 0.3)

    # GATE 1: Hard Constraints (High Risk + Contradiction)
    # If it's a High Risk claim and the Judge found a contradiction, we kill it immediately.
    if risk == "High" and "contradict" in status:
        return 0, f"Hard Constraint Violation: {key_point}"

    # GATE 2: Strong Soft Contradiction
    # If it's Medium/Low Risk but the evidence is overwhelming (>60%), we kill it.
    if risk != "High" and "contradict" in status and confidence > 0.6:
        return 0, f"Strong Thematic Contradiction: {key_point}"

    # GATE 3: Evidence Accumulation (The "Acceptance" Gate)
    if "consistent" in status:
        # Dynamic Threshold Adjustment
        req_conf = TH_CONSISTENCY
        if risk == "High": req_conf += 0.1  # Requires 0.65
        if risk == "Low": req_conf -= 0.1   # Requires 0.45
        
        if confidence >= req_conf:
            return 1, f"Validated by Evidence: {key_point}"

    # GATE 4: Uncertainty Resolution (Conservative Bias)
    # If confidence is too low (< 30%), we reject it as unproven.
    if confidence < TH_UNCERTAINTY:
        return 0, "Insufficient evidence (Conservative Bias)."
    
    # If the Judge said "Contradict" but it was weak (didn't trigger Gate 1/2),
    # we still default to 0 because we prioritize narrative safety.
    if "contradict" in status:
        return 0, f"Contradicted by evidence: {key_point}"
        
    # Final Catch-all (should rarely be reached)
    return 1, f"Plausible within constraints: {key_point}"