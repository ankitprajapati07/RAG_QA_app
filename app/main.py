import os
from typing import List

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from .embeddings import get_openai_client
from .retriever import Document, FaissRetriever
from .generator import generate_answer
from .schemas import AskRequest, AskResponse

# ---------------------------
# Config & Environment
# ---------------------------
load_dotenv()
INDEX_DIR = os.getenv("INDEX_DIR", "./index")

os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="RAG Q&A API", version="1.0.0")
retriever = FaissRetriever(index_dir=INDEX_DIR)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
	if not req.question.strip():
		raise HTTPException(status_code=400, detail="Question must not be empty")
	try:
		results = retriever.query(req.question, top_k=5)
	except FileNotFoundError:
		raise HTTPException(status_code=500, detail="Index not found. Run ingestion first.")

	if not results:
		raise HTTPException(status_code=404, detail="No matches found")

	top_doc, _ = results[0]
	contexts: List[str] = []
	for doc, _score in results[:3]:
		contexts.append(f"Question: {doc.question}\nAnswer: {doc.description}")

	generated = generate_answer(req.question, contexts)
	return AskResponse(id=top_doc.id, question=top_doc.question, generated_response=generated)
