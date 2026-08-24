import os
import pandas as pd
from groq import Groq
from .embeddings import embed_query

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)

def metadata_filter(df, category=None, max_price_bdt=None, brand=None):
    mask = pd.Series(True, index=df.index)
    if category:
        mask &= df["category"].str.lower() == category.lower()
    if max_price_bdt:
        mask &= df["price_bdt"] <= max_price_bdt
    if brand:
        mask &= df["brand"].str.lower() == brand.lower()
    return df[mask]

def retrieve(query_text, index, df, k=5, category=None, max_price_bdt=None, brand=None):
    if df.empty:
        return df
        
    qvec = embed_query(query_text)
    
    # Retrieve a larger candidate pool, then apply metadata filters
    search_k = min(k * 4, len(df))
    if search_k == 0:
        return df
        
    _, idxs = index.search(qvec, search_k)
    
    # Filter out -1 indices which FAISS returns if not enough elements
    valid_idxs = [i for i in idxs[0] if i != -1]
    
    if not valid_idxs:
        return pd.DataFrame(columns=df.columns)
        
    candidates = df.iloc[valid_idxs]
    candidates = metadata_filter(candidates, category, max_price_bdt, brand)
    
    return candidates.head(k)

def build_prompt(query_text, retrieved):
    context_lines = []
    for _, r in retrieved.iterrows():
        chunk_text = str(r['chunk_text'])[:200] if pd.notna(r['chunk_text']) else ""
        context_lines.append(
            f"- {r['brand']} {r['model']} | {r['category']} | {r['processor']} | "
            f"{r['ram_gb']}GB RAM | {r['price_bdt']} BDT | {chunk_text}"
        )
    context = "\\n".join(context_lines)
    
    return f\"\"\"You are a device recommendation assistant. Only use the CONTEXT below -- do not invent devices, specs, or prices that are not listed.

CONTEXT:
{context}

USER QUERY: {query_text}

Recommend the best matching device(s) from the CONTEXT above, with a short justification tied to the user's stated needs. If nothing in CONTEXT fits well, say so honestly instead of guessing.\"\"\"

def rag_recommend(query_text, index, df, k=5, category=None, max_price_bdt=None, brand=None):
    retrieved = retrieve(query_text, index, df, k, category, max_price_bdt, brand)
    
    if retrieved.empty:
        return {
            "answer": "I couldn't find any devices matching your criteria in the database.",
            "retrieved": retrieved,
            "prompt": ""
        }
        
    prompt = build_prompt(query_text, retrieved)
    
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    
    return {"answer": answer, "retrieved": retrieved, "prompt": prompt}
