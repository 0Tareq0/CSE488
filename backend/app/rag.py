import os
import pandas as pd
from groq import Groq
from .embeddings import embed_query

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key is not configured.")
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

from .constraints import extract_constraints

def build_prompt(query_text, retrieved):
    context_lines = []
    for _, r in retrieved.iterrows():
        chunk_text = str(r['chunk_text'])[:200] if pd.notna(r['chunk_text']) else ""
        context_lines.append(
            f"- {r['brand']} {r['model']} | {r['category']} | {r['processor']} | "
            f"{r['ram_gb']}GB RAM | {r['price_bdt']} BDT | {chunk_text}"
        )
    context = "\n".join(context_lines)
    
    return f"""You are a device recommendation assistant.

You may ONLY recommend devices contained in the RETRIEVED DEVICES section.

Never invent a device, specification, price, benchmark, feature, or review.

If the retrieved data does not support a claim, do not make that claim.

Use the user's query to compare the retrieved devices.

Return:
- recommended device
- price in BDT
- key reasons
- relevant specifications
- brief comparison when useful

The retrieved devices are authoritative.

RETRIEVED DEVICES:
{context}

USER QUERY: {query_text}"""

def rag_recommend(query_text, index, df, k=5, category=None, max_price_bdt=None, brand=None):
    client = get_groq_client()
    
    # Extract constraints if they are not explicitly provided
    constraints = extract_constraints(query_text, client, GROQ_MODEL)
    
    # Use explicit arguments if provided (e.g. from UI filters), otherwise fallback to extracted
    final_category = category or constraints.get("category")
    final_max_price = max_price_bdt or constraints.get("max_price_bdt")
    final_brand = brand or constraints.get("brand")
    
    retrieved = retrieve(query_text, index, df, k, final_category, final_max_price, final_brand)
    
    if retrieved.empty:
        return {
            "answer": "No matching device was found in the dataset.",
            "retrieved": retrieved,
            "prompt": ""
        }
        
    prompt = build_prompt(query_text, retrieved)
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    
    return {"answer": answer, "retrieved": retrieved, "prompt": prompt}
