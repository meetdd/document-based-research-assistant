import os
from parser import process_pdf
from db import save_to_chroma, voter_collection

def index_all_pdfs():
    folder = "uploaded_pdfs"
    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    
    if not files:
        print("No PDFs found in 'uploaded_pdfs' folder.")
        return

    total_saved = 0
    for file in files:
        file_path = os.path.join(folder, file)
        print(f"\nProcessing: {file}")
        
        # 1. Extract data using our OCR parser
        chunks = process_pdf(file_path, file)
        print(f"Extracted {len(chunks)} records. Vectorizing and saving to ChromaDB...")
        
        # 2. Convert to embeddings and save to database
        saved_count = save_to_chroma(chunks)
        total_saved += saved_count
        
    print(f"\n=== INDEXING COMPLETE ===")
    print(f"Total new records saved: {total_saved}")
    print(f"Total records now in Database: {voter_collection.count()}")

if __name__ == "__main__":
    index_all_pdfs()