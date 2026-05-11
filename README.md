# BOLTYN – Turn documents into answers

**Boltyn** is an AI-powered document assistant that answers questions about your PDF documents – fully local, no cloud API required.  
Uses **Ollama** (LLaMA, Qwen, etc.) + **ChromaDB** for Retrieval-Augmented Generation (RAG).

## Features

- 📄 **PDF Indexing** – Upload PDFs or place them in `docs/`
- 🔍 **RAG Queries** – Ask questions, the AI answers from your documents
- 🌐 **Bilingual** – UI and answers in German or English (switchable)
- 🎤 **Voice Mode** – Ask via microphone, listen to answers (browser)
- 🐳 **Docker** – Full stack (Ollama + Boltyn) via `docker-compose`
- 🧠 **Model Switching** – Change LLMs on the fly in the UI
- ⚙️ **RAG Parameters** – Score-threshold, top-k, temperature live adjustable

## Prerequisites

- **Python 3.11+** (for local setup)
- **Ollama** (local or in Docker network)
- **~2–20 GB RAM** depending on LLM (qwen2.5:32b ~20 GB, qwen2.5:14b ~9 GB, llama3.2:3b ~2 GB)

## Quick Start

### 1. Install Ollama

**Locally (macOS / Linux):**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Or via Docker:**

```bash
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
```

### 2. Pull a model

```bash
# Recommended (good for RAG)
ollama pull qwen2.5:14b

# Lighter (for weaker hardware)
ollama pull llama3.2:3b

# Embedding model (required for document indexing)
ollama pull nomic-embed-text
```

### 3. Setup Boltyn

```bash
# Clone the repo
git clone https://github.com/your-user/boltyn.git
cd boltyn

# Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Edit .env (e.g. OLLAMA_MODEL)
```

### 4. Start

```bash
python server.py
```

Open **http://localhost:8000/admin** in your browser.

### 5. Index documents

Place PDFs in `docs/` and click **"Learn now"** in the admin UI.  
Or upload PDFs directly via drag & drop.

## Docker Full Setup

```bash
# Start Ollama + Boltyn
docker-compose up -d

# Pull a model into Ollama
docker exec -it boltyn-ollama ollama pull qwen2.5:14b
docker exec -it boltyn-ollama ollama pull nomic-embed-text

# Open UI
open http://localhost:8000/admin
```

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:32b` | Active LLM |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB storage path |
| `SERVER_HOST` | `0.0.0.0` | Server host |
| `SERVER_PORT` | `8000` | Server port |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status` | Server status |
| `POST` | `/api/chat` | Ask a question |
| `GET` | `/api/models` | Available models |
| `POST` | `/api/models/switch` | Switch active model |
| `GET` | `/api/config` | Current configuration |
| `GET` | `/api/documents` | Indexed documents |
| `POST` | `/api/documents/toggle` | Enable/disable document |
| `POST` | `/api/documents/delete` | Delete document |
| `POST` | `/api/ingest` | Start indexing |
| `POST` | `/api/reindex` | Rebuild database |
| `POST` | `/api/upload` | Upload PDF |
| `GET` | `/api/log` | Indexing log |
| `GET/POST` | `/api/params` | RAG parameters |
| `POST` | `/api/restart` | Restart server |
| `GET` | `/admin` | Admin UI |

## Project Structure

```
boltyn/
├── server.py         # FastAPI server (API + Admin UI)
├── rag_engine.py     # RAG engine (ChromaDB + LLM)
├── ingest.py         # PDF ingest pipeline
├── config.py         # Configuration (.env)
├── static/
│   └── admin.html    # Admin frontend (single file)
├── docs/             # Place PDFs here
├── chroma_db/        # Vector database (local)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md         # English
└── READMEDE.md       # Deutsch
```

## License

MIT – see [LICENSE](LICENSE).
