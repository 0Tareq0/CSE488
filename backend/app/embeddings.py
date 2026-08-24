import os
import httpx
import numpy as np
from fastapi import HTTPException

def get_embedding_model():
    # Model weights are now loaded via API, so this is a no-op, 
    # but kept for compatibility with the startup event
    pass

def embed_query(query_text: str):
    hf_token = os.getenv("HF_TOKEN", "")
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    with httpx.Client() as client:
        response = client.post(
            api_url, 
            headers=headers, 
            json={"inputs": query_text, "options": {"wait_for_model": True}},
            timeout=10.0
        )
        
        if response.status_code != 200:
            print("HF API Error:", response.text)
            # Fallback to zero vector if HF API is rate limited
            return np.zeros((1, 384)).astype("float32")
            
        v = np.array(response.json()).astype("float32")
        return v.reshape(1, -1)
