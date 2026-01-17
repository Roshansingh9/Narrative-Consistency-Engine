# Narrative Consistency via Constraint-Based Multi-Agent Reasoning
### Kharagpur Data Science Hackathon 2026 — Track A


**Architecture:** Deterministic Constraint Satisfaction over Vector Memory

---

## 0. Executive Summary
This system is **not** a generative LLM wrapper. It is a **Binary Decision Engine** designed to answer one specific question: *Can a proposed backstory logically exist within the established narrative world?*

Unlike naive RAG approaches which ask "Is this plausible?", this architecture:
1. Treats narrative facts as **Immutable Constraints**.
2. Decomposes reasoning into **Adversarial Agents** (Prosecutor vs. Advocate).
3. Enforces a **4-Stage Deterministic Gate** to decide the final label.
4. Operates with a **Conservative Bias**: If evidence is insufficient, the system rejects the claim to preserve narrative integrity.

---

## 1. System Architecture

The pipeline strictly follows a 6-Layer design where no component exceeds its authority.

### **Layer 1: Narrative Memory (Pathway)**
* **Technology:** Pathway (Real-time data processing).
* **Logic:** Sliding Window Chunking (512 words, 50 overlap) to capture causal chains rather than just keywords.
* **Output:** A persisted Vector Index on Port 8001.

### **Layer 2 & 3: Claim Normalization (Taxonomy)**
* **Goal:** Assign a Risk Tier to every claim.
* **High Risk (Red):** Temporal, Physical, Existence. (Strict enforcement).
* **Medium Risk (Orange):** Ideological, Relational. (Strong evidence required).
* **Low Risk (Yellow):** Psychological, Symbolic. (Lenient enforcement).

### **Layer 4: Evidence Retrieval**
* **Method:** Dense Vector Retrieval (MiniLM-L6-v2).
* **Scope:** Retrieves Top-5 relevant narrative chunks per claim.

### **Layer 5: Multi-Agent Reasoning**
Instead of a single "Check Consistency" prompt, we instantiate three agents:
1. **The Prosecutor:** Aggressively hunts for contradictions (Time errors, Dead characters alive).
2. **The Advocate:** Constructs the most plausible defense using narrative gaps.
3. **The Judge:** Evaluates the *strength* of the evidence (0.0 to 1.0), not the final label.

### **Layer 6: Aggregation (The Global Brain)**
The final decision is **Deterministic**, not probabilistic.
1. **Gate 1 (Hard Constraint):** If Risk is High and Contradiction found → `0` (Immediate Kill).
2. **Gate 2 (Soft Constraint):** If Risk is Medium and Contradiction is strong (>60%) → `0`.
3. **Gate 3 (Evidence):** Does confidence exceed the `consistency_threshold` (0.45)? → `1`.
4. **Gate 4 (Uncertainty):** If confidence < `uncertainty_threshold` (0.20) → `0`.

---

## 2. Installation & Setup

### Prerequisites
* Python 3.10+
* Groq API Key (or OpenAI equivalent)
* Pathway

### Installation
```bash
# 1. Clone/Navigate
cd project

# 2. Install Dependencies
pip install -r requirements.txt
```

### Configuration
All hyperparameters are central in `config.yaml`.

* `consistency_threshold`: Lower this to increase Recall (catch more True claims).
* `uncertainty_threshold`: Lower this to reduce Conservative Bias.
* `port`: Default is 8001 for Memory Server.

---

## 3. Usage Guide

### Phase A: Start the Memory Server
This must run in a separate terminal. It ingests the books and serves vector embeddings.

```bash
python pathway_pipeline/index.py
```

Wait for: "Narrative Memory Server starting on 0.0.0.0:8001"

### Phase B: Optimization (Training)
We use a dedicated script to mathematically solve for the optimal thresholds using the training data.

```bash
python project/optimize_thresholds.py
```

* **Logic:** Performs a Grid Search on 80% of train.csv.
* **Output:** Automatically updates config.yaml with the parameters that maximize Accuracy.

### Phase C: Run Inference (Generate Result)
This script processes test.csv, runs the multi-agent debate, and applies the aggregation logic.

```bash
python run_inference.py
```

* **Input:** `data/test.csv`
* **Output:** `Result.csv` (Root directory)

---

## 4. Tuning & Methodology

We employed Grid Search Optimization on an 80/20 Stratified Split of the training data.

**Critical Finding:** Initially, the system was too conservative , rejecting valid claims because the evidence wasn't "perfect."

**Adjustment:** The optimizer identified that lowering `consistency_threshold` from 0.70 → 0.45 maximized F1-Score.

**Result:** This allows "plausible" claims (where the Advocate makes a good point) to pass, while still catching hard contradictions.

---

## 5. File Structure

```
project/
├── data/
│   ├── Books/          # Raw .txt files
│   ├── train.csv       # Development Data
│   └── test.csv        # Final Evaluation Data
├── pathway_pipeline/
│   ├── ingest.py       # Data Loading
│   └── index.py        # Vector Server
├── retrieval/
│   └── retrieve.py     # Connection to Pathway
├── reasoning/
│   ├── claims.py       # Data Objects
│   ├── normalization.py# Taxonomy Classifier
│   ├── debate.py       # Prosecutor/Advocate/Judge Agents
│   └── aggregation.py  # Deterministic Logic Gates
├── optimize_thresholds.py # Training/Grid Search Script
├── run_inference.py    # Main Orchestrator
├── config.yaml         # Central Control Plane
└── README.md           # This file
```

---

## 6. License & Acknowledgments

Built for Kharagpur Data Science Hackathon 2026.

* **Engine:** Pathway
* **LLM Provider:** Groq (Llama-3)
* **Embeddings:** SentenceTransformers (MiniLM)
