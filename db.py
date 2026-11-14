import chromadb
from chromadb.utils import embedding_functions

# Initialize persistent ChromaDB so data isn't lost on restart
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Multilingual embedding model handles cross-lingual search (English query -> Hindi text)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Create or connect to the collection
voter_collection = chroma_client.get_or_create_collection(
    name="electoral_roll_db",
    embedding_function=embed_fn
)

def save_to_chroma(chunks: list):
    if not chunks:
        print("No chunks provided to save.")
        return 0

    ids = [item["id"] for item in chunks]
    docs = [item["text"] for item in chunks]
    metas = [item["metadata"] for item in chunks]

    # Insert in batches to prevent memory crashes
    batch_limit = 300
    for i in range(0, len(ids), batch_limit):
        voter_collection.upsert(
            ids=ids[i : i + batch_limit],
            documents=docs[i : i + batch_limit],
            metadatas=metas[i : i + batch_limit]
        )
    
    return len(ids)

if __name__ == "__main__":
    print(f"ChromaDB initialized. Current records in DB: {voter_collection.count()}")