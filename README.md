# RAG Q&A API (FAISS + OpenAI)

A simple FastAPI app that answers medical questions using your dataset. It finds the most relevant entries and generates dynamic responses using OpenAI.

## Quick Start

### 1. Setup Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
. .venv\Scripts\Activate.ps1

# If activation fails, run this first:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
. .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file in project root:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-mini
INDEX_DIR=./index
```

### 3. Run the API
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API
```powershell
# PowerShell
$body = @{ question = "What does CSF lymphocytosis with normal glucose suggest?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post -Body $body -ContentType "application/json"

# Or visit http://127.0.0.1:8000/docs for interactive testing
```

## Project Structure
```
.
├── app/
│   ├── embeddings.py     # OpenAI embeddings
│   ├── retriever.py      # FAISS search
│   ├── generator.py      # LLM answer generation
│   ├── schemas.py        # API models
│   ├── ingest.py         # Build index from dataset
│   └── main.py           # FastAPI app
├── dataset/
│   └── qa_model_dataset.json
├── index/                # Pre-built FAISS index (included)
├── requirements.txt
└── README.md
```

## API Usage

**Endpoint:** `POST /ask`

**Request:**
```json
{
  "question": "Your medical question here"
}
```

**Response:**
```json
{
  "id": "5",
  "question": "A 43-year-old Asian man presents with...",
  "generated_response": "Based on the CSF findings..."
}
```

## How It Works

1. **Retrieval**: Finds top 5 most similar dataset entries using FAISS
2. **Context**: Uses top 3 entries as context for the LLM
3. **Generation**: LLM creates a dynamic answer based on retrieved context
4. **Response**: Returns closest match ID, question, and generated answer

## Notes

- The FAISS index is pre-built and included in the repo
- Uses `text-embedding-3-small` for embeddings
- Uses `gpt-5-mini` for answer generation
- Retrieval uses cosine similarity on normalized vectors

## License
MIT