import json
import os

STORE_PATH = "official_summaries.json"

def save_summary(pdf_name: str, summary: dict):
    data = {}
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    data[pdf_name] = summary
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_all_summaries() -> dict:
    if not os.path.exists(STORE_PATH):
        return {}
    with open(STORE_PATH, encoding="utf-8") as f:
        return json.load(f)