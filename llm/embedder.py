# project/llm/embedder.py

import logging
from sentence_transformers import SentenceTransformer
import pathway as pw
import numpy as np

# Global variable to hold the model (Lazy Loading)
_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _get_model():
    """Singleton pattern to load model once per worker."""
    global _MODEL
    if _MODEL is None:
        logging.info(f"Loading embedding model: {_MODEL_NAME}")
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL

@pw.udf
def embed_text(text: str) -> np.ndarray:
    """
    Standalone Pathway UDF for embedding text.
    """
    if not text:
        # Return empty vector of correct dimension (384 for MiniLM)
        return np.zeros(384)
        
    model = _get_model()
    # Return numpy array as expected by Pathway
    return np.array(model.encode(text))