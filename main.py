import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from parser import process_pdf
from db import save_to_chroma, voter_collection
from rag_engine import ask_question, compute_roll_stats

app = FastAPI(title="Electoral Roll AI Research Assistant")

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok", "records_in_db": voter_collection.count()}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Indexing uploaded file: {file.filename}")
    chunks = process_pdf(file_path, file.filename)
    saved_count = save_to_chroma(chunks)

    return {
        "filename": file.filename,
        "records_extracted": len(chunks),
        "records_saved": saved_count,
        "total_records_in_db": voter_collection.count(),
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = ask_question(request.question)
    return QuestionResponse(answer=answer)


@app.get("/stats")
def stats():
    result = compute_roll_stats()
    if not result:
        raise HTTPException(status_code=404, detail="No documents indexed yet.")
    return result