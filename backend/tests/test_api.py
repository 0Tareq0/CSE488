from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import pytest
import os
from unittest.mock import patch

@pytest.fixture(autouse=True)
def setup_dummy_data():
    import app.faiss_index
    # Create small dummy dataset
    df = pd.DataFrame([
        {
            "device_id": 1,
            "category": "Mobile",
            "brand": "TestBrand",
            "model": "TestModel X",
            "processor": "TestCPU",
            "gpu": None,
            "ram_gb": 8,
            "storage_gb": 128,
            "display": "6 inch",
            "battery": "4000",
            "price_bdt": 50000.0,
            "price_currency": "BDT",
            "source_dataset": "test",
            "chunk_text": "A great test phone",
            "embedding": np.random.rand(384).tolist()
        }
    ])
    app.faiss_index._df_cache = df
    app.faiss_index._index_cache = {}
    
    with patch("app.main.get_embedding_model"):
        yield

from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_devices():
    response = client.get("/devices")
    assert response.status_code == 200
    data = response.json()
    assert "devices" in data
    assert len(data["devices"]) == 1
    assert data["devices"][0]["brand"] == "TestBrand"

@patch("app.rag.get_groq_client")
@patch("app.rag.embed_query")
def test_recommend(mock_embed, mock_get_groq):
    mock_embed.return_value = np.random.rand(1, 384).astype("float32")
    # Mock groq response
    mock_client = mock_get_groq.return_value
    mock_response = mock_client.chat.completions.create.return_value
    mock_response.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': 'I recommend TestBrand TestModel X.'})})]

    response = client.post(
        "/recommend", 
        json={"query": "test phone", "category": "Mobile"},
        params={"index_type": "Flat"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "retrieved_devices" in data
    assert len(data["retrieved_devices"]) > 0
