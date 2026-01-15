from llm.wrapper import query_llm

def run_debate(claim, evidence_chunks):
    """
    Orchestrates the debate between Believer and Skeptic.
    """
    # 1. Prepare Evidence String
    context_text = "\n\n".join([
        f"Source ({chunk['metadata']['book_name']}): {chunk['text']}" 
        for chunk in evidence_chunks
    ])

    print(f"   --> Analyzing {len(evidence_chunks)} evidence chunks...")

    # --- AGENT 1: THE BELIEVER ---
    believer_sys = "You are a logical assistant. Your goal is to support the given claim using ONLY the provided context."
    believer_prompt = f"""
    CLAIM: "{claim}"
    
    CONTEXT:
    {context_text}
    
    TASK: Provide a concise argument supporting this claim. If the context supports it, quote the text.
    """
    believer_arg = query_llm(believer_sys, believer_prompt)


    # --- AGENT 2: THE SKEPTIC ---
    skeptic_sys = "You are a critical assistant. Your goal is to disprove or find contradictions to the claim using ONLY the provided context."
    skeptic_prompt = f"""
    CLAIM: "{claim}"
    
    CONTEXT:
    {context_text}
    
    TASK: Provide a concise argument contradicting this claim. Look for details that don't match.
    """
    skeptic_arg = query_llm(skeptic_sys, skeptic_prompt)

    return believer_arg, skeptic_arg

def judge_verdict(claim, believer_arg, skeptic_arg):
    """
    The Judge reviews the arguments and issues a final verdict.
    """
    judge_sys = "You are an impartial Judge. You evaluate arguments based on evidence."
    judge_prompt = f"""
    I need you to determine if a claim is Consistent or Contradicted based on the following debate.
    
    CLAIM: "{claim}"
    
    ARGUMENT FOR (Believer):
    {believer_arg}
    
    ARGUMENT AGAINST (Skeptic):
    {skeptic_arg}
    
    VERDICT FORMAT:
    Status: [Consistent/Contradicted/Unknown]
    Confidence: [0-100]%
    Reasoning: [One sentence summary]
    """
    
    return query_llm(judge_sys, judge_prompt)