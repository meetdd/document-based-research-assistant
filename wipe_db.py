from db import voter_collection

def wipe_database():
    existing_docs = voter_collection.get()
    ids = existing_docs.get("ids", [])
    
    if ids:
        print(f"Found {len(ids)} old records. Deleting...")
        voter_collection.delete(ids=ids)
        print("Database is now completely empty.")
    else:
        print("Database is already empty.")

if __name__ == "__main__":
    wipe_database()