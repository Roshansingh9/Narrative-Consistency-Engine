# project/pathway_pipeline/index.py

import sys
import os
import yaml
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathway_pipeline.ingest import get_book_source
from llm.embedder import embed_text 

# Load Configuration
config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# --- Custom Components ---

class BookSplitter:
    """
    A custom parser + splitter combined.
    1. Decodes the binary data to text.
    2. Splits the text into chunks (windows of words).
    3. Returns a list of (text_chunk, metadata) tuples.
    """
    def __init__(self, chunk_size=400, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __call__(self, data):
        # 1. Safe Decode
        try:
            text = data.decode("utf-8")
        except Exception:
            return []

        # 2. Split by words (Simple but effective for books)
        words = text.split()
        chunks = []
        
        if not words:
            return []

        # Sliding window approach
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            # We return a tuple: (content, extra_metadata)
            chunks.append((chunk_text, {}))
            
        return chunks

# --- UDFs (Helper Functions) ---

@pw.udf
def make_metadata(meta_wrapper) -> dict:
    """
    Accepts the raw Json metadata object and returns a clean Python dict.
    """
    if not meta_wrapper:
        return {"book_name": "unknown"}
    
    try:
        raw_path = meta_wrapper["path"]
        path_str = str(raw_path).strip('"').strip("'")
        base = os.path.basename(path_str)
        book_name = base.replace(".txt", "")
        
        return {
            "path": path_str,
            "book_name": book_name
        }
    except Exception:
        return {"book_name": "unknown"}

def run_memory_server():
    print("--- Initializing Narrative Memory Layer ---")

    # 1. Initialize Data Source
    data_dir = config['pathway']['data_dir']
    print(f"Watching directory: {data_dir}")
    if not os.path.exists(data_dir):
         os.makedirs(data_dir, exist_ok=True)
         
    raw_files = get_book_source(data_dir)

    # 2. Transform Data
    documents = raw_files.select(
        data=pw.this.data,
        _metadata=make_metadata(pw.this._metadata)
    )

    # 3. Build Vector Store
    # We use our Custom 'BookSplitter' which handles both parsing and splitting.
    print("Building Index...")
    vector_server = VectorStoreServer(
        documents,
        embedder=embed_text,
        parser=BookSplitter(chunk_size=400, chunk_overlap=50) 
    )

    # 4. Run Server
    host = config['pathway']['host']
    port = config['pathway']['port']
    print(f"Narrative Memory Server starting on {host}:{port}")
    
    vector_server.run_server(
        host=host,
        port=port,
        threaded=True,
        with_cache=True
    )

if __name__ == "__main__":
    run_memory_server()