# chroma_rag — Session-Based RAG with Django + ChromaDB

A production-aware **Retrieval-Augmented Generation (RAG)** backend built with Django, ChromaDB, LangChain, and OpenAI.
Designed with one goal: **user gets an answer in 2–5 seconds.**

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

─────────────────────────────────────────────────

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
chroma_rag/
├── services/
│   ├── document_loader.py   # Loads PDF, DOCX, TXT, Excel files
│   ├── chunking.py          # Recursive text chunking (chunk_size=1000, overlap=200)
│   ├── embedding.py         # HuggingFace embeddings (all-MiniLM-L6-v2) — loaded once
│   ├── vectordb.py          # ChromaDB — create per-session store, load existing store
│   ├── retrieval.py         # Dense similarity retrieval from ChromaDB
│   ├── hybrid_search.py     # BM25 + Dense search merged with RRF
│   ├── reranker.py          # Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2)
│   ├── llm.py               # gpt-3.5-turbo — answer generation + question condensing
│   └── session_store.py     # DB-backed chat history (get / add / clear)
├── models.py                # ChatSession + ChatMessage Django models
├── views.py                 # All API views
├── urls.py                  # URL routing
└── requirements.txt         # App-level dependencies
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chroma/upload/` | Upload a document — returns `session_id` |
| `GET`  | `/api/chroma/upload/` | List all past sessions |
| `POST` | `/api/chroma/query/`  | Ask a question — requires `session_id` |
| `GET`  | `/api/chroma/session/<session_id>/` | Get full chat history for a session |
| `POST` | `/api/chroma/clear-session/` | Clear chat messages for a session |

### Upload Request
```
POST /api/chroma/upload/
Content-Type: multipart/form-data
Body: file=<your document>
```

### Query Request
```json
POST /api/chroma/query/
{
    "query": "What is the refund policy?",
    "session_id": "uuid-returned-from-upload"
}
```

---

## Session Architecture

Every uploaded document creates a completely isolated session:

```
chroma_rag/documents/<session_id>/filename.pdf   ← document stored here
chroma_sessions/<session_id>/                    ← chroma vector DB stored here
DB: ChatSession row                              ← links session_id to both paths
DB: ChatMessage rows                             ← all Q&A turns for this session
```

This means:
- Old sessions always work — their document and vector DB are never deleted
- New uploads never overwrite old ones
- User can switch between sessions and get correct answers from the right document
- Chat history persists across server restarts (stored in DB, not memory)

---

## Latency Fixes Applied

The original pipeline took **9–12 seconds per query**. Here is what was causing it and how it was fixed:

| Problem | Fix | Time Saved |
|---------|-----|------------|
| PDF reload + rechunking on every query | Cache chunks in memory after upload | ~1–3s |
| BM25 index rebuilt on every query | Build once, store in `_rag_cache` | ~0.5–1s |
| Query rewriter LLM call on every query | Removed entirely | ~0.8–1.5s |
| ChromaDB re-initialized on every query | Kept alive in `_rag_cache` | ~0.3–0.5s |
| Cross-Encoder scoring 25 docs | Reduced hybrid top_k 25 → 10 | ~0.3–0.7s |
| **Result** | **Target latency: 2–5 seconds** | ✅ |

---

## Models Used

| Model | Purpose | Cost |
|-------|---------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings | Free (local) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking | Free (local) |
| `gpt-3.5-turbo` | Question condensing + Answer generation | ~$0.001/query |

---

## Database Models

### ChatSession
Stores one row per uploaded document session.
```
session_id    — unique UUID, returned to frontend on upload
document_name — original filename shown in UI
document_path — full path to the uploaded file on disk
chroma_path   — path to this session's ChromaDB vector store
created_at    — timestamp
```

### ChatMessage
Stores every Q&A turn for a session.
```
session    — foreign key to ChatSession
role       — "user" or "assistant"
content    — the message text
created_at — timestamp (messages ordered oldest first)
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...        # Required
DJANGO_SECRET_KEY=...        # Required
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r chroma_rag/requirements.txt

# 2. Run migrations
python manage.py makemigrations chroma_rag
python manage.py migrate

# 3. Start server
python manage.py runserver
```

---

## Known Issues

| Issue | Status |
|-------|--------|
| Migrations must be run manually before first query | Pending automation |
| In-process cache holds only the last active session — switching between two sessions rapidly causes cold reloads | Acceptable for now |
| No authentication — any user can access any session by guessing session_id | Needs auth layer before production |

---

## Remaining Work / Next Steps

| Feature | Priority | Notes |
|---------|----------|-------|
| Connect the UI (frontend) | 🔴 High | Templates exist, API is ready |
| Authentication / user login | 🔴 High | Sessions are currently open |
| Session deletion endpoint | 🟡 Medium | Delete session + its document + chroma DB |
| Multi-session cache | 🟡 Medium | Cache last N sessions instead of just the last one |
| Evaluation framework | 🟢 Low | Measure retrieval quality (precision, recall) |
| Multi-query retrieval | 🟢 Low | Already built in `rag/` app — can be ported when latency allows |
| Query rewriting | 🟢 Low | Already built — disabled for latency, re-enable with async if needed |
