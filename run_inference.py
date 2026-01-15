import pandas as pd
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(__file__))

from reasoning.claims import NarrativeClaim
from reasoning.normalization import normalize_claim
from retrieval.retrieve import retrieve_evidence
from reasoning.debate import run_debate
from reasoning.aggregation import aggregate_decision

TEST_CSV = os.path.join(os.path.dirname(__file__), "data/test.csv")
OUTPUT_CSV = "results.csv"

def main():
    
    
    # Handle CSV location flexibility
    input_path = TEST_CSV
    if not os.path.exists(input_path):
        if os.path.exists("test.csv"): input_path = "test.csv"
        elif os.path.exists("project/data/test.csv"): input_path = "project/data/test.csv"
        else:
            print(" Error: test.csv not found!")
            return

    print(f"Loading Input: {input_path}")
    df = pd.read_csv(input_path)
    results = []

    for idx, row in df.iterrows():
        # 1. Claim Generation
        claim = NarrativeClaim(row['content'], row['book_name'], row['char'], row['id'])
        print(f"\nProcessing {idx+1}/{len(df)}: ID {claim.id}")

        # 2. Normalization
        claim = normalize_claim(claim)
        
        # 3. Retrieval
        claim = retrieve_evidence(claim)
        
        # 4. Multi-Agent Reasoning
        claim = run_debate(claim)
        
        # 5. Aggregation
        pred, rationale = aggregate_decision(claim)
        
        print(f"   -> Verdict: {pred} ({rationale[:60]}...)")
        
        results.append({
            "id": claim.id,
            "prediction": pred,
            "rationale": rationale
        })

    # Save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n Submission Generated: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()