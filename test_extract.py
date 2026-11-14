import os
import pymupdf

folder = "uploaded_pdfs"
files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

pdf_path = os.path.join(folder, files[0])
print("Reading file:", pdf_path)

doc = pymupdf.open(pdf_path)
print("Total pages:", len(doc))

# Check page 3 (index 2) where voter boxes start
page = doc[2]
blocks = page.get_text("blocks")

for i, b in enumerate(blocks[:10]):
    print(f"\n--- BLOCK {i} ---")
    print(b[4].strip())