# SessionBasedRAG — Django + ChromaDB

A production-aware **Retrieval-Augmented Generation (RAG)** backend built with Django, ChromaDB, LangChain, and OpenAI.
Designed with one goal: **user gets an answer in 2–5 seconds.**

---

## What This Project Does

- User uploads a document (PDF, DOCX, TXT, Excel)
- System indexes it into ChromaDB and creates a unique session
- User asks questions — system retrieves relevant chunks and answers using GPT
- Every Q&A turn is saved to the database
- User can come back to old sessions and continue the conversation — answers still work because each session has its own document and vector store

---

## Architecture

```
User Uploads Document
        ↓
Load + Chunk Document         (PyPDF / Docx / TXT / Excel)
        ↓
Embed Chunks                  (all-MiniLM-L6-v2 — free, local)
        ↓
Store in ChromaDB             (per-session isolated vector DB)
        ↓
Build BM25 Index              (once, cached in memory)
        ↓
Session Created in DB         (session_id returned to frontend)

─────────────────────────────────────────────────────────────

User Sends Query + session_id
        ↓
Load Session from DB          (get document path + chroma path)
        ↓
Hit In-Process Cache          (vector_store + chunks + bm25 — no disk reload)
        ↓
Condense Question             (only if chat history exists — gpt-3.5-turbo)
        ↓
Hybrid Search                 (Dense ChromaDB + BM25 keyword — RRF merged)
        ↓
Cross-Encoder Rerank          (ms-marco-MiniLM — top 10 → best 5)
        ↓
Generate Answer               (gpt-3.5-turbo — answers from context only)
        ↓
Save Q&A to DB                (persists across server restarts)
        ↓
Answer returned to user
```

---

## Project Structure

```
SessionBasedRAG/
├── chroma_rag/
│   ├── services/
│   │   ├── document_loader.py   # Loads PDF, DOCX, TXT, Excel
│   │   ├── chunking.py          # Recursive text chunking (size=1000, overlap=200)
│   │   ├── embedding.py         # HuggingFace embeddings — loaded once at startup
│   │   ├── vectordb.py          # ChromaDB — create per-session store + load old ones
│   │   ├── retrieval.py         # Dense similarity retrieval from ChromaDB
│   │   ├── hybrid_search.py     # BM25 + Dense merged with Reciprocal Rank Fusion
│   │   ├── reranker.py          # Cross-Encoder reranking (top 10 → best 5)
│   │   ├── llm.py               # GPT-3.5-turbo — answer generation + question condensing
│   │   └── session_store.py     # DB-backed chat history (get / add / clear)
│   ├── migrations/
│   │   └── 0001_initial.py      # DB migration for ChatSession + ChatMessage
│   ├── models.py                # ChatSession + ChatMessage Django models
│   ├── views.py                 # All API views
│   ├── urls.py                  # URL routing
│   ├── apps.py
│   └── requirements.txt         # All dependencies for this app
├── rag_backend/
│   ├── settings.py              # Django project settings
│   ├── urls.py                  # Root URL config
│   └── wsgi.py
├── manage.py
├── requirements.txt             # Root level dependencies
└── .env.example                 # Copy this to .env and fill in your keys
```

---

## Quick Start — Step by Step

### Step 1 — Clone the repo

```bash
git clone https://github.com/Jo9gi/SessionBasedRAG.git
cd SessionBasedRAG
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r chroma_rag/requirements.txt
```

### Step 4 — Create your .env file

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...your-openai-key-here...
DJANGO_SECRET_KEY=your-django-secret-key-here
```

### Step 5 — Run database migrations

```bash
python manage.py makemigrations chroma_rag
python manage.py migrate
```

> This creates the `ChatSession` and `ChatMessage` tables in the SQLite database.
> You must do this before the first query or you will get a database error.

### Step 6 — Start the server

```bash
python manage.py runserver
```

Server starts at `http://localhost:8000`

---

## API Usage — Step by Step

### Step 1 — Upload a document

```
POST http://localhost:8000/api/chroma/upload/
Content-Type: multipart/form-data
Body: file = <your document>
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.xlsx`, `.xls`

Response:
```json
{
    "status": "success",
    "message": "Document uploaded and indexed. Use the session_id for all queries.",
    "session_id": "3f2a1b4c-...",
    "document_name": "policy.pdf",
    "total_chunks": 42
}
```

> Save the `session_id` — you need it for every query.

---

### Step 2 — Ask a question

```
POST http://localhost:8000/api/chroma/query/
Content-Type: application/json
```

```json
{
    "query": "What is the refund policy?",
    "session_id": "3f2a1b4c-..."
}
```

Response:
```json
{
    "session_id": "3f2a1b4c-...",
    "document_name": "policy.pdf",
    "query": "What is the refund policy?",
    "answer": "Refunds are accepted within 30 days of purchase..."
}
```

---

### Step 3 — Ask a follow-up question (chat history kicks in)

```json
{
    "query": "What if I paid by card?",
    "session_id": "3f2a1b4c-..."
}
```

> The system automatically detects the chat history and condenses
> "What if I paid by card?" into "What is the refund policy for card payments?"
> before searching — so follow-up questions work correctly.

---

### Step 4 — View all past sessions (sidebar)

```
GET http://localhost:8000/api/chroma/upload/
```

Response:
```json
{
    "sessions": [
        {"session_id": "3f2a1b4c-...", "document_name": "policy.pdf", "created_at": "..."},
        {"session_id": "9d8e7f6a-...", "document_name": "invoice.pdf", "created_at": "..."}
    ]
}
```

---

### Step 5 — Open an old session and load its chat history

```
GET http://localhost:8000/api/chroma/session/3f2a1b4c-.../
```

Response:
```json
{
    "session_id": "3f2a1b4c-...",
    "document_name": "policy.pdf",
    "messages": [
        {"role": "user",      "content": "What is the refund policy?", "created_at": "..."},
        {"role": "assistant", "content": "Refunds within 30 days...",  "created_at": "..."}
    ]
}
```

---

### Step 6 — Clear chat history for a session

```
POST http://localhost:8000/api/chroma/clear-session/
Content-Type: application/json
```

```json
{
    "session_id": "3f2a1b4c-..."
}
```

> This deletes the conversation only. The document and vector DB remain intact.
> User can still ask new questions in the same session.

---

## All API Endpoints Summary

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `POST` | `/api/chroma/upload/` | Upload document → get `session_id` |
| `GET`  | `/api/chroma/upload/` | List all past sessions |
| `POST` | `/api/chroma/query/`  | Ask a question (requires `session_id`) |
| `GET`  | `/api/chroma/session/<session_id>/` | Get full chat history for a session |
| `POST` | `/api/chroma/clear-session/` | Clear chat messages (keeps document) |

---

## Session Architecture — How Old Sessions Still Work

Every uploaded document gets its own isolated storage:

```
chroma_rag/documents/<session_id>/filename.pdf   ← document saved here (never deleted)
chroma_sessions/<session_id>/                    ← ChromaDB vector store for this session
DB: ChatSession row                              ← links session_id → document + chroma paths
DB: ChatMessage rows                             ← all Q&A turns for this session
```

When a user opens an old session:
1. System loads `ChatSession` from DB → gets the document path and chroma path
2. Loads that session's ChromaDB vector store from disk
3. Reloads and re-chunks the original document for BM25
4. Caches everything in memory for fast follow-up queries
5. Loads the full chat history from `ChatMessage` table

---

## Latency Fixes — How We Got from 9–12s to 2–5s

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| PDF reload + rechunking on every query | BM25 had no index cache | Build BM25 once on upload, store in `_rag_cache` |
| ChromaDB re-initialized on every query | New `Chroma()` instance per request | Keep vector_store alive in `_rag_cache` |
| Query rewriter LLM call on every query | Extra OpenAI API round trip | Removed entirely |
| Cross-Encoder scoring 25 docs | top_k was set too high | Reduced hybrid top_k `25 → 10`, rerank to `5` |
| **Before** | **9–12 seconds** | |
| **After** | **2–5 seconds** | ✅ |

---

## Models Used

| Model | Purpose | Cost |
|-------|---------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings (local) | Free |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking (local) | Free |
| `gpt-3.5-turbo` | Question condensing + Answer generation | ~$0.001/query |

> First run downloads HuggingFace models (~100MB). Subsequent runs use local cache.

---

## Database Models

### ChatSession
One row per uploaded document. Created automatically on upload.
```
session_id    — unique UUID (returned to frontend)
document_name — original filename
document_path — full path to the file on disk
chroma_path   — path to this session's ChromaDB vector store
created_at    — upload timestamp
```

### ChatMessage
One row per message turn (user or assistant). Ordered oldest first.
```
session    — foreign key → ChatSession
role       — "user" or "assistant"
content    — the message text
created_at — message timestamp
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `DJANGO_SECRET_KEY` | ✅ Yes | Django secret key |

---

## Known Issues

| Issue | Notes |
|-------|-------|
| Migrations must be run manually on first setup | Run `python manage.py migrate` before first query |
| In-process cache holds only the last active session | Switching sessions rapidly causes a cold reload from disk (~2–3s extra once) |
| No authentication | Any user can query any session by knowing the `session_id` — needs auth before production |

---

## Remaining Work / Next Steps

| Feature | Priority | Notes |
|---------|----------|-------|
| Connect the UI (frontend) | 🔴 High | Templates exist, API is fully ready |
| Authentication / user login | 🔴 High | Tie sessions to logged-in users |
| Session deletion endpoint | 🟡 Medium | Delete session + document + chroma DB from disk |
| Multi-session memory cache | 🟡 Medium | Cache last N sessions instead of only the most recent |
| Evaluation framework | 🟢 Low | Measure retrieval quality (precision, recall, RAGAS) |
| Multi-query retrieval | 🟢 Low | Already built in old repo — port when latency budget allows |
| Query rewriting | 🟢 Low | Already built — disabled for latency, can re-enable with async |
