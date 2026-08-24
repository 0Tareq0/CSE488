import os
os.environ["HF_HOME"] = "/tmp"

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # Same model used in the notebook
        EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedding_model

def embed_query(query_text: str):
    model = get_embedding_model()
    v = model.encode([query_text])[0].astype("float32")
    return v.reshape(1, -1)
