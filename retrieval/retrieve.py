import requests
import yaml
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load Config
config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception:
    # Fallback default
    config = {'pathway': {'host': '0.0.0.0', 'port': 8000}}

SERVER_URL = f"http://{config['pathway']['host']}:{config['pathway']['port']}/v1/retrieve"

def retrieve_evidence(claim_obj, k=5):
    """
    Queries the Pathway server for context relevant to the claim.
    Updates the claim_obj.evidence field.
    """
    if not claim_obj.content:
        return claim_obj

    # Construct query: "BookName: ClaimText"
    query_text = f"{claim_obj.book_name}: {claim_obj.content}"
    
    payload = {
        "query": query_text,
        "k": k
    }
    
    try:
        # FIX: Increased timeout to 60 seconds to prevent read timeouts
        response = requests.post(SERVER_URL, json=payload, timeout=800)
        response.raise_for_status()
        results = response.json()
        
        # Store evidence in the claim object
        claim_obj.evidence = results
        
    except Exception as e:
        print(f"   [!] Retrieval Error for Claim {claim_obj.id}: {e}")
        claim_obj.evidence = []
        
    return claim_obj