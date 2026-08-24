import json

def extract_constraints(query_text, client, model):
    prompt = f"""Extract constraints from the user query. Return a JSON object with the following keys, setting them to null if not specified in the query:
- "category": "Mobile" or "Laptop" or null
- "max_price_bdt": number or null (extract any budget/price limits)
- "brand": string or null

User query: "{query_text}"
Return ONLY raw JSON, no markdown blocks or extra text."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {}
