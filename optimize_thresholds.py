import pandas as pd
import numpy as np
import os
import sys
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure project root is in path
sys.path.append(os.path.dirname(__file__))

from reasoning.claims import NarrativeClaim
from reasoning.normalization import normalize_claim
from retrieval.retrieve import retrieve_evidence
from reasoning.debate import run_debate

# Config
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "data/train.csv")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def map_label(label_str):
    label = label_str.lower().strip()
    return 1 if "consistent" in label else 0

def simulate_aggregation(row, th_consist, th_uncertain):
    """
    Simulates the Aggregation Layer logic mathematically without re-running LLMs.
    """
    status = row['status']
    confidence = row['confidence']
    risk = row['risk']

    # GATE 1: Hard Constraints
    if risk == "High" and "contradict" in status:
        return 0

    # GATE 2: Strong Soft Contradiction
    if risk != "High" and "contradict" in status and confidence > 0.6:
        return 0

    # GATE 3: Evidence Accumulation
    if "consistent" in status:
        req_conf = th_consist
        if risk == "High": req_conf += 0.1
        if risk == "Low": req_conf -= 0.1
        
        if confidence >= req_conf:
            return 1

    # GATE 4: Uncertainty
    if confidence < th_uncertain:
        return 0
    
    if "contradict" in status:
        return 0
        
    return 1

def run_optimization():
    print("--- 🧠 Training Aggregation Parameters (Grid Search) ---")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    df['target'] = df['label'].apply(map_label)
    
    # 2. Split 80/20 (We optimize ONLY on the 80% Train set)
    train_df, val_df = train_test_split(df, test_size=0.5, random_state=42, stratify=df['target'])
    print(f"Optimization Set: {len(train_df)} samples")

    # 3. Pre-compute LLM outputs (The slow part)
    print("Generating LLM Scores (this happens once)...")
    cache_data = []
    
    for idx, row in train_df.iterrows():
        print(f"\rProcessing {idx+1}/{len(train_df)}...", end="")
        
        # Run Pipeline
        claim = NarrativeClaim(row['content'], row['book_name'], row['char'], row['id'])
        claim = normalize_claim(claim)
        claim = retrieve_evidence(claim)
        claim = run_debate(claim)
        
        # Extract Raw Signals
        verdict = claim.judge_verdict
        cache_data.append({
            "target": row['target'],
            "status": verdict.get("status", "uncertain").lower(),
            "confidence": float(verdict.get("confidence", 0.0)),
            "risk": claim.risk_tier
        })
    
    print("\n\n--- Running Grid Search ---")
    
    # 4. Grid Search
    best_acc = 0.0
    best_params = (0.7, 0.4) # Defaults
    
    # Test Consistency Thresholds: 0.40 to 0.90
    for th_c in np.arange(0.4, 0.95, 0.05):
        # Test Uncertainty Thresholds: 0.10 to 0.60
        for th_u in np.arange(0.1, 0.65, 0.05):
            
            preds = [simulate_aggregation(row, th_c, th_u) for row in cache_data]
            targets = [row['target'] for row in cache_data]
            
            acc = accuracy_score(targets, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_params = (float(th_c), float(th_u))
                # print(f"New Best: Acc {acc:.2%} (Consist={th_c:.2f}, Uncert={th_u:.2f})")

    print("="*40)
    print(f"🏆 BEST PARAMETERS FOUND")
    print(f"Training Accuracy: {best_acc:.2%}")
    print(f"Consistency Threshold: {best_params[0]:.2f}")
    print(f"Uncertainty Threshold: {best_params[1]:.2f}")
    print("="*40)

    # 5. Update Config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    config['aggregation']['consistency_threshold'] = best_params[0]
    config['aggregation']['uncertainty_threshold'] = best_params[1]
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)
        
    print(f"✅ Updated {CONFIG_PATH} with optimized values.")

if __name__ == "__main__":
    run_optimization()