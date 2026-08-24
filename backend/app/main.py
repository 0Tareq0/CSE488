from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from .embeddings import get_embedding_model
from .faiss_index import load_or_build_index, get_device_df
from .rag import rag_recommend
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Device Recommender API")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    query: str
    category: Optional[str] = None
    max_price_bdt: Optional[float] = None
    brand: Optional[str] = None
    k: Optional[int] = 5

class RecommendResponse(BaseModel):
    answer: str
    retrieved_devices: List[dict]
    latency_ms: float

# Load resources on startup
@app.on_event("startup")
def startup_event():
    # Pre-load embedding model and index
    get_embedding_model()
    load_or_build_index()
    get_device_df()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, index_type: str = Query("HNSW")):
    try:
        import time
        t0 = time.time()
        
        index = load_or_build_index(index_type)
        df = get_device_df()
        
        result = rag_recommend(
            req.query, 
            index, 
            df, 
            k=req.k, 
            category=req.category, 
            max_price_bdt=req.max_price_bdt, 
            brand=req.brand
        )
        
        latency_ms = (time.time() - t0) * 1000
        
        # Convert DataFrame to list of dicts for JSON serialization
        # Replace NaN with None
        retrieved_list = result["retrieved"].replace({float('nan'): None}).to_dict(orient="records")
        
        return RecommendResponse(
            answer=result["answer"],
            retrieved_devices=retrieved_list,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices")
def list_devices(page: int = 1, page_size: int = 20):
    df = get_device_df()
    start = (page - 1) * page_size
    end = start + page_size
    devices = df.iloc[start:end].replace({float('nan'): None}).to_dict(orient="records")
    return {"devices": devices, "total": len(df)}

@app.get("/api/devices/{device_id}")
def get_device(device_id: int):
    df = get_device_df()
    device = df[df["device_id"] == device_id]
    if device.empty:
        raise HTTPException(status_code=404, detail="Device not found")
    return device.replace({float('nan'): None}).to_dict(orient="records")[0]
