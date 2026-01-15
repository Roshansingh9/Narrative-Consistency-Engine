import json
import os
import yaml
from llm.wrapper import query_llm

# Load Taxonomy from Config
config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    TAXONOMY = config.get('taxonomy', {})
except Exception:
    # Fallback default if config isn't ready
    TAXONOMY = {
        "high_risk": ["Temporal", "Physical", "Existence", "Identity"],
        "medium_risk": ["Ideological", "Relational", "Political"],
        "low_risk": ["Psychological", "Cultural", "Symbolic"]
    }

def get_risk_tier(category):
    """Maps a category to its risk tier based on config."""
    category = category.strip().title()
    if category in TAXONOMY.get('high_risk', []):
        return "High"
    if category in TAXONOMY.get('medium_risk', []):
        return "Medium"
    return "Low"

def normalize_claim(claim_obj):
    """
    Analyzes the claim content and assigns a Category and Risk Tier.
    """
    system_prompt = """
    You are a Narrative Taxonomist. Classify the user's backstory claim into EXACTLY ONE of these categories:
    
    [High Risk]
    - Temporal: Claims about dates, timelines, or sequence of events.
    - Physical: Claims about location, items held, injuries, or physical capabilities.
    - Existence: Claims about whether a character/object existed or died.
    - Identity: Claims about names, lineage, or roles.

    [Medium Risk]
    - Ideological: Claims about beliefs, religion, or political stance.
    - Relational: Claims about who knew whom, marriages, or feuds.

    [Low Risk]
    - Psychological: Claims about feelings, motivations, or thoughts.
    - Cultural: Claims about rituals, customs, or habits.

    RESPONSE FORMAT: JSON only.
    {
        "category": "CategoryName",
        "reasoning": "Brief explanation"
    }
    """

    user_prompt = f"""
    Book: {claim_obj.book_name}
    Character: {claim_obj.character_name}
    Claim: "{claim_obj.content}"
    """

    try:
        response = query_llm(system_prompt, user_prompt)
        # Naive JSON parsing (Robustness can be added)
        data = json.loads(response.replace("```json", "").replace("```", ""))
        
        category = data.get("category", "Symbolic")
        claim_obj.classification = category
        claim_obj.risk_tier = get_risk_tier(category)
        
    except Exception as e:
        print(f"Normalization Error: {e}")
        claim_obj.classification = "Symbolic"
        claim_obj.risk_tier = "Low"

    return claim_obj