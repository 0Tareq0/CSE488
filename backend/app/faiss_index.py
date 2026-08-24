import os
import faiss
import pandas as pd
import numpy as np

# Use an environment variable for the parquet path to allow local and docker testing
PARQUET_PATH = os.getenv("PARQUET_PATH", os.path.join(os.path.dirname(__file__), "../../notebook/output/devices_embeddings.parquet"))

_df_cache = None
_index_cache = {}

def get_device_df():
    global _df_cache
    if _df_cache is None:
        try:
            _df_cache = pd.read_parquet(PARQUET_PATH)
        except Exception as e:
            # Create a small dummy dataframe so the app can start even if parquet isn't built yet
            print(f"Warning: Could not load {PARQUET_PATH}, starting with empty dataframe: {e}")
            _df_cache = pd.DataFrame(columns=[
                "device_id", "category", "brand", "model", "processor", "gpu", 
                "ram_gb", "storage_gb", "display", "battery", "price_bdt", 
                "price_currency", "source_dataset", "chunk_text", "embedding"
            ])
    return _df_cache

def load_or_build_index(index_type: str = "HNSW"):
    """
    Build (or load from cache) a FAISS index.
    Defaults to HNSW because it offers a great balance of low latency and high recall.
    """
    global _index_cache
    
    if index_type in _index_cache:
        return _index_cache[index_type]
        
    df = get_device_df()
    if df.empty:
        # Dummy index
        return faiss.IndexFlatL2(384)
        
    embedding_matrix = np.array(df["embedding"].tolist(), dtype="float32")
    dim = embedding_matrix.shape[1]
    
    if index_type == "Flat":
        idx = faiss.IndexFlatL2(dim)
        idx.add(embedding_matrix)
    elif index_type == "IVF":
        quantizer = faiss.IndexFlatL2(dim)
        nlist = min(50, len(df))
        idx = faiss.IndexIVFFlat(quantizer, dim, nlist)
        idx.train(embedding_matrix)
        idx.add(embedding_matrix)
        idx.nprobe = 8
    elif index_type == "IVF+PQ":
        quantizer = faiss.IndexFlatL2(dim)
        nlist = min(50, len(df))
        m = 8
        bits = 8
        idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
        idx.train(embedding_matrix)
        idx.add(embedding_matrix)
        idx.nprobe = 8
    else:  # Default to HNSW
        # HNSW is chosen as the default because it provides very low latency
        # with minimal recall tradeoff, suitable for our dataset size.
        idx = faiss.IndexHNSWFlat(dim, 32)
        idx.hnsw.efConstruction = 80
        idx.add(embedding_matrix)
        
    _index_cache[index_type] = idx
    return idx
