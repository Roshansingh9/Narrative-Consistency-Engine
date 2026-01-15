import uuid

class NarrativeClaim:
    """
    Represents a single narrative unit to be verified.
    """
    def __init__(self, content, book_name, character_name, claim_id=None):
        self.id = claim_id or str(uuid.uuid4())
        self.content = content
        self.book_name = book_name
        self.character_name = character_name
        
        # Populated by Normalization Layer
        self.classification = "Unknown"
        self.risk_tier = "Unknown"
        
        # Populated by Reasoning Layer
        self.evidence = []
        self.prosecutor_arg = ""
        self.defense_arg = ""
        self.judge_verdict = ""
        
    def __repr__(self):
        return f"<Claim {self.id[:4]}: [{self.risk_tier}] {self.content[:30]}...>"