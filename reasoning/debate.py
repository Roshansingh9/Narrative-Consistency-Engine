import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.wrapper import query_llm

def run_debate(claim_obj):
    """
    Orchestrates the 3-Agent Debate.
    """
    print(f"   ⚖️  Debating Claim: [{claim_obj.risk_tier}] {claim_obj.content[:40]}...")

    # 1. Prepare Context
    if not claim_obj.evidence:
        context_text = "NO DIRECT EVIDENCE FOUND IN NARRATIVE MEMORY."
    else:
        # Deduplicate and format
        seen = set()
        unique_chunks = []
        for item in claim_obj.evidence:
            text = item.get('text', '').strip()
            if text and text not in seen:
                seen.add(text)
                unique_chunks.append(f"Source ({item.get('metadata', {}).get('book_name', 'Unknown')}): {text}")
        context_text = "\n\n".join(unique_chunks)

    # --- AGENT 1: PROSECUTOR ---
    prosecutor_prompt = f"""
    SYSTEM: You are the Prosecutor. Your goal is to DISPROVE the claim.
    CLAIM: "{claim_obj.content}"
    CONTEXT:
    {context_text}
    TASK: Attack this claim. Find contradictions, timeline errors, or impossibilities.
    """
    claim_obj.prosecutor_arg = query_llm("You are a ruthless Prosecutor.", prosecutor_prompt)

    # --- AGENT 2: ADVOCATE ---
    defense_prompt = f"""
    SYSTEM: You are the Advocate. Your goal is to DEFEND the claim.
    CLAIM: "{claim_obj.content}"
    CONTEXT:
    {context_text}
    TASK: Defend this claim. Show plausibility or fit within the character's history.
    """
    claim_obj.defense_arg = query_llm("You are a strategic Defense Attorney.", defense_prompt)

    # --- AGENT 3: LOCAL JUDGE ---
    judge_prompt = f"""
    SYSTEM: You are an impartial Judge.
    CLAIM: "{claim_obj.content}"
    
    [PROSECUTION]: {claim_obj.prosecutor_arg}
    [DEFENSE]: {claim_obj.defense_arg}
    
    TASK: Return a JSON verdict.
    {{ "status": "Consistent" OR "Contradicted" OR "Uncertain", "confidence": 0.0 to 1.0, "key_point": "One sentence summary" }}
    """
    
    try:
        response = query_llm("You are a Judge. Output JSON only.", judge_prompt)
        # Clean JSON
        clean_json = response.replace("```json", "").replace("```", "").strip()
        # Handle cases where LLM adds extra text
        if "{" in clean_json:
            clean_json = clean_json[clean_json.find("{"):clean_json.rfind("}")+1]
            
        claim_obj.judge_verdict = json.loads(clean_json)
    except Exception as e:
        print(f"   [!] Judge Error: {e}")
        claim_obj.judge_verdict = {"status": "Uncertain", "confidence": 0.0, "key_point": "Error parsing verdict"}

    return claim_obj