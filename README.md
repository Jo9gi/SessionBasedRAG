# RAG_Jango — Advanced RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** backend built with Django, Milvus, LangChain and OpenAI.

---

## Architecture

```text
User Query
    ↓
Query Rewriter        (gpt-3.5-turbo — expands vague queries)
    ↓
Multi Query Generator (4 query variants for better recall)
    ↓
Hybrid Search         (Dense Search via Milvus + BM25 keyword search)
    ↓
Reciprocal Rank Fusion (merges results from all queries)
    ↓
Cross-Encoder Reranker (ms-marco-MiniLM — reranks top 20 → best 5)
    ↓
LLM Generation        (gpt-4.1-mini — answers from context only)
    ↓
Answer
```

---

## Project Structure

```
RAG_Jango/
├── rag/
│   ├── documents/          # Place your PDF files here
│   ├── services/
│   │   ├── document_loader.py   # PDF loading + metadata enrichment
│   │   ├── chunking.py          # Recursive text chunking
│   │   ├── embedding.py         # HuggingFace embeddings (all-MiniLM-L6-v2)
│   │   ├── vectordb.py          # Milvus vector store (HNSW + Cosine)
│   │   ├── retrieval.py         # Dense similarity retrieval
│   │   ├── hybrid_search.py     # BM25 + Dense search with RRF
│   │   ├── query_rewriter.py    # Query expansion (gpt-3.5-turbo)
│   │   ├── multi_query.py       # Multi query retrieval + RRF
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── llm.py               # OpenAI answer generation
│   │   └── test_rag.py          # Interactive CLI pipeline
│   └── ...
├── rag_backend/            # Django project settings
├── Dockerfile
├── docker-compose.yml      # Django + Milvus services
├── requirements.txt
└── .env                    # Your secrets (never committed)
```

---

## RAG Phases Implemented

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Metadata Filtering (source, page, file_name) | ✅ |
| 2 | Query Rewriting (gpt-3.5-turbo) | ✅ |
| 3 | Multi Query Retrieval + RRF | ✅ |
| 4 | Cross-Encoder Reranking | ✅ |
| 5 | Hybrid Search (Dense + BM25) | ✅ |
| 6 | Evaluation Framework | 🔜 |
| 7 | Multi PDF Support | 🔜 |
| 8 | Chat History | 🔜 |
| 9 | REST API Layer | 🔜 |

---

## Quick Start with Docker (Recommended)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/RAG_Jango.git
cd RAG_Jango
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...your-key-here...
DJANGO_SECRET_KEY=your-django-secret-key
```

### 3. Add your PDF

Place your PDF inside `rag/documents/`.

### 4. Start the containers

```bash
docker compose up --build
```

This starts:
- **Django** on `http://localhost:8000`
- **Milvus** on `localhost:19530`

### 5. Run the RAG pipeline

```bash
docker compose exec web python -m rag.services.test_rag
```

---

## Quick Start without Docker

### 1. Create virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Linux / Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Milvus separately

Follow the [Milvus standalone install guide](https://milvus.io/docs/install_standalone-docker.md) or use Docker just for Milvus:

```bash
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.4.4 milvus run standalone
```

### 4. Apply migrations and run

```bash
python manage.py migrate
python manage.py runserver
```

### 5. Run the RAG pipeline

```bash
python -m rag.services.test_rag
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...          # Required — OpenAI API key
DJANGO_SECRET_KEY=...          # Required — Django secret key
HF_TOKEN=                      # Optional — HuggingFace token for private models
MILVUS_URI=http://localhost:19530  # Optional — defaults to localhost
```

> `.env` is gitignored and will never be committed.

---

## Models Used

| Model | Purpose | Cost |
|-------|---------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings | Free (local) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking | Free (local) |
| `gpt-3.5-turbo` | Query rewriting | ~$0.001/query |
| `gpt-4.1-mini` | Answer generation | ~$0.01/query |

---

## Notes

- First run downloads HuggingFace models (~100MB). Subsequent runs use cache.
- Milvus data is persisted in a Docker volume — vectors survive container restarts.
- `chunk_size=1000`, `chunk_overlap=200` works well for most PDFs.
- Multi query generates 4 variants per query — expect 4x more Milvus calls.
