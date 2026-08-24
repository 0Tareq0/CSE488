import os, re, math
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = "./notebook/data"
PARQUET_OUT = "./api/data/devices_embeddings.parquet"
os.makedirs(os.path.dirname(PARQUET_OUT), exist_ok=True)

FX_TO_BDT = {
    "BDT": 1.0, "USD": 122.0, "EUR": 132.0, "GBP": 155.0,
    "INR": 1.46, "PKR": 0.43, "CNY": 16.8, "AED": 33.2,
}

def to_bdt(amount, currency):
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return None
    currency = (currency or "BDT").upper().strip()
    rate = FX_TO_BDT.get(currency)
    if rate is None:
        return None
    return round(float(amount) * rate, 2)

def parse_money(text):
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None, None
    s = str(text)
    currency = None
    m = re.match(r"\s*([A-Za-z]{3})\s", s)
    if m:
        currency = m.group(1).upper()
    if "\u09f3" in s or "TK" in s.upper():
        currency = currency or "BDT"
    num = re.sub(r"[^\d.]", "", s)
    if num == "":
        return currency, None
    try:
        return currency, float(num)
    except ValueError:
        return currency, None

def read_csv_robust(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def parse_ram(text):
    if pd.isna(text): return None
    m = re.search(r"(\d+)\s*GB", str(text), re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_storage(text):
    if pd.isna(text): return None
    s = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", s, re.IGNORECASE)
    if m: return float(m.group(1)) * 1024
    m = re.search(r"(\d+(?:\.\d+)?)\s*GB", s, re.IGNORECASE)
    return float(m.group(1)) if m else None

def make_text_blob(row, extra_fields):
    parts = [f"{k}: {v}" for k, v in extra_fields.items() if pd.notna(v) and str(v).strip() != ""]
    return f"{row.get('brand','')} {row.get('model','')} — " + "; ".join(parts)

records = []

# 1) laptop_public.csv
try:
    lp = read_csv_robust(f"{DATA_DIR}/Orginal_public_csv/laptop_public.csv")
    for _, r in lp.iterrows():
        price_bdt = to_bdt(r.get("Price_euros"), "EUR")
        extra = {"Type": r.get("TypeName"), "Screen": r.get("ScreenResolution"),
                 "CPU": r.get("Cpu"), "RAM": r.get("Ram"), "Storage": r.get("Memory"),
                 "GPU": r.get("Gpu"), "OS": r.get("OpSys"), "Weight": r.get("Weight")}
        records.append({
            "category": "Laptop", "brand": r.get("Company"), "model": r.get("Product"),
            "processor": r.get("Cpu"), "gpu": r.get("Gpu"),
            "ram_gb": parse_ram(r.get("Ram")), "storage_gb": parse_storage(r.get("Memory")),
            "display": f"{r.get('Inches')} inch {r.get('ScreenResolution')}", "battery": None,
            "price_bdt": price_bdt, "price_currency": "EUR", "price_original": r.get("Price_euros"),
            "source_dataset": "laptop_public.csv",
            "description": make_text_blob(r, extra), "review_text": None,
        })
except Exception as e: print("Skipping laptop_public:", e)

# 2) mobile_public.csv
try:
    mp = read_csv_robust(f"{DATA_DIR}/Orginal_public_csv/mobile_public.csv")
    for _, r in mp.iterrows():
        cur, amt = parse_money(r.get("Launched Price (USA)"))
        cur = cur or "USD"
        records.append({
            "category": "Mobile", "brand": r.get("Company Name"), "model": r.get("Model Name"),
            "processor": r.get("Processor"), "gpu": None,
            "ram_gb": parse_ram(r.get("RAM")), "storage_gb": None,
            "display": r.get("Screen Size"), "battery": r.get("Battery Capacity"),
            "price_bdt": to_bdt(amt, cur), "price_currency": cur, "price_original": amt,
            "source_dataset": "mobile_public.csv",
            "description": make_text_blob(r, {"Weight": r.get("Mobile Weight"), "RAM": r.get("RAM"), "Front Camera": r.get("Front Camera"), "Back Camera": r.get("Back Camera"), "Processor": r.get("Processor"), "Battery": r.get("Battery Capacity"), "Screen": r.get("Screen Size"), "Year": r.get("Launched Year")}), "review_text": None,
        })
except Exception as e: print("Skipping mobile_public:", e)

# 3) enhanced
try:
    for fname, category in [("public_mobile_enhanced.csv", "Mobile"), ("public_laptop_enhanced.csv", "Laptop")]:
        df = read_csv_robust(f"{DATA_DIR}/public_dataset_enhanched/{fname}")
        for _, r in df.iterrows():
            records.append({
                "category": category, "brand": r.get("brand"), "model": r.get("model"),
                "processor": r.get("processor"), "gpu": r.get("gpu"),
                "ram_gb": r.get("ram_gb"), "storage_gb": r.get("storage_gb"),
                "display": f"{r.get('display_size_inches')} inch {r.get('display_resolution')}",
                "battery": r.get("battery_mah") if pd.notna(r.get("battery_mah")) else r.get("battery_wh"),
                "price_bdt": to_bdt(r.get("price_original"), r.get("price_currency")), "price_currency": r.get("price_currency"),
                "price_original": r.get("price_original"),
                "source_dataset": fname,
                "description": r.get("description"), "review_text": r.get("review_text"),
            })
except Exception as e: print("Skipping enhanced:", e)

# 4) scraped
try:
    for fname, category in [("Scraped_New_Mobile.csv", "Mobile"), ("Scraped_New_laptop.csv", "Laptop")]:
        df = read_csv_robust(f"{DATA_DIR}/new_device_scraped/{fname}")
        for _, r in df.iterrows():
            cur, amt = parse_money(r.get("price_bdt"))
            cur = cur or "BDT"
            records.append({
                "category": category, "brand": r.get("brand"), "model": r.get("model"),
                "processor": r.get("processor"), "gpu": r.get("gpu"),
                "ram_gb": parse_ram(r.get("ram_gb")), "storage_gb": parse_storage(r.get("storage_gb")),
                "display": r.get("display"), "battery": r.get("battery"),
                "price_bdt": to_bdt(amt, cur), "price_currency": cur, "price_original": amt,
                "source_dataset": fname,
                "description": r.get("project_description"), "review_text": r.get("user_review"),
            })
except Exception as e: print("Skipping scraped:", e)

unified = pd.DataFrame.from_records(records)
unified.insert(0, "device_id", range(1, len(unified) + 1))
unified = unified.fillna({"description": "", "review_text": ""})
unified["chunk_text"] = unified["description"].astype(str) + " " + unified["review_text"].astype(str)
unified["chunk_text"] = unified["chunk_text"].str.strip()
unified = unified[unified["chunk_text"] != ""]
unified = unified.drop_duplicates(["category", "brand", "model", "price_bdt"])

# Fix PyArrow mixed type errors for object columns
for col in ["processor", "gpu", "display", "battery", "price_currency", "source_dataset"]:
    unified[col] = unified[col].astype(str).replace("nan", None)

print(f"Generated {len(unified)} records. Computing embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(unified["chunk_text"].tolist(), batch_size=32, show_progress_bar=True)
unified["embedding"] = [e.tolist() for e in embeddings]

unified.to_parquet(PARQUET_OUT, index=False)
print(f"Successfully saved to {PARQUET_OUT}")
