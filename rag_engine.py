import os
import re
from collections import Counter
from dotenv import load_dotenv
from groq import Groq
from db import voter_collection
from official_summary_store import load_all_summaries

# 1. Load the API key securely from the .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Set the Groq model
GROQ_MODEL = "openai/gpt-oss-20b"

def compute_roll_stats():
    all_records = voter_collection.get(include=["metadatas"])
    metas = all_records.get("metadatas", [])
    if not metas:
        return None

    official = load_all_summaries()
    if official:
        total_count = sum(s.get("net_total", 0) for s in official.values())
        male_count = sum(s.get("net_male", 0) for s in official.values())
        female_count = sum(s.get("net_female", 0) for s in official.values())
        stat_source = "Official ECI summary tables (certified, printed in each PDF)"
    else:
        gender_counts = Counter(m.get("gender", "Unknown") for m in metas)
        total_count = len(metas)
        male_count = gender_counts.get("Male", 0)
        female_count = gender_counts.get("Female", 0)
        stat_source = "Extracted voter records only — no official summary found"

    ratio_str = f"{male_count}:{female_count}"
    females_per_1000_males = round((female_count / male_count) * 1000, 1) if male_count else 0

    surname_counts = Counter(m.get("surname", "Unknown") for m in metas)
    top_surnames = dict(surname_counts.most_common(20))
    pdf_sources = sorted(set(m.get("source_pdf", "") for m in metas))

    return {
        "total_voters": total_count,
        "male": male_count,
        "female": female_count,
        "ratio": ratio_str,
        "per_1000": females_per_1000_males,
        "surnames": top_surnames,
        "sources": pdf_sources,
        "extracted_record_count": len(metas),   # for transparency vs. official total
        "stat_source": stat_source,
    }

def find_voter_by_name(name_query: str):
    all_records = voter_collection.get(include=["metadatas"])
    metas = all_records.get("metadatas", [])
    q = name_query.strip()
    return [m for m in metas if q and q in (m.get("voter_name") or "")]

def ask_question(user_question: str):
    q_clean = user_question.lower()

    find_match = re.search(r'find\s+(.+?)\s+voter', user_question, re.IGNORECASE)
    stat_keywords = ["how many", "count", "surname", "ratio", "male", "female", "total"]

    if find_match:
        print("Routing to Exact Name Lookup...")
        name_query = find_match.group(1).strip()
        hits = find_voter_by_name(name_query)
        if not hits:
            return f"No voter matching '{name_query}' found in the indexed records."
        lines = [
            f"- {m.get('voter_name')} | Age {m.get('age')} | {m.get('gender')} | "
            f"Page {m.get('page')} of {m.get('source_pdf')}"
            for m in hits
        ]
        context_str = "Exact matches from database:\n" + "\n".join(lines)

    elif any(k in q_clean for k in stat_keywords):
        print("Routing to Aggregation Engine...")
        stats = compute_roll_stats()
        if not stats:
            return "No documents uploaded yet."
        context_str = (
            f"Statistics source: {stats['stat_source']}\n"
            f"- Total Voters (official): {stats['total_voters']}\n"
            f"- Gender Breakdown -> Male: {stats['male']} | Female: {stats['female']}\n"
            f"- Ratio: {stats['ratio']} ({stats['per_1000']} females per 1000 males)\n"
            f"- Surname Distribution (from {stats['extracted_record_count']} extracted records, approximate): {stats['surnames']}\n"
            f"- Source Files: {stats['sources']}"
        )
    else:
        print("Routing to Vector Search Engine...")
        results = voter_collection.query(query_texts=[user_question], n_results=10)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        context_lines = [
            f"[Source: {meta.get('source_pdf','?')}, Page: {meta.get('page','?')}] {doc_text}"
            for doc_text, meta in zip(docs, metas)
        ]
        context_str = "\n".join(context_lines)

    prompt = (
        "You are an AI research assistant analyzing electoral roll documents. "
        "Answer the user's question strictly using the provided context below. "
        "If the context notes that surname counts are approximate, say so in your answer. "
        "If quoting specific voters, include the Page number and Source PDF.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {user_question}\nAnswer:"
    )

    print("Generating answer instantly via Groq LPUs...")
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API Error: {e}"

# Simple CLI loop to test the questions
if __name__ == "__main__":
    print("\nAI Research Assistant Ready! (Type 'exit' to quit)")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        answer = ask_question(query)
        print("\n--- ANSWER ---")
        print(answer)