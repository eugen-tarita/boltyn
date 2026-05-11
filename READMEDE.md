# BOLTYN – Turn documents into answers

**Boltyn** ist ein KI-gestützter Dokumenten-Assistent, der Fragen zu deinen PDF-Dokumenten beantwortet – vollständig lokal, ohne Cloud-API.  
Nutzt **Ollama** (LLaMA, Qwen, etc.) + **ChromaDB** für Retrieval-Augmented Generation (RAG).

## Features

- 📄 **PDF-Indizierung** – Lade PDFs hoch oder lege sie in `docs/` ab
- 🔍 **RAG-Abfragen** – Stelle Fragen, die KI antwortet aus deinen Dokumenten
- 🌐 **Zweisprachig** – UI und Antworten auf Deutsch oder Englisch (umschaltbar)
- 🎤 **Sprachmodus** – Frage per Mikrofon, Antwort per Sprachausgabe (Browser)
- 🐳 **Docker** – Kompletter Stack (Ollama + Boltyn) per `docker-compose`
- 🧠 **Modellwahl** – Wechsel zwischen verschiedenen LLMs live im UI
- ⚙️ **RAG-Parameter** – Score-Threshold, Top-K, Temperature live einstellbar

## Voraussetzungen

- **Python 3.11+** (für lokalen Betrieb)
- **Ollama** (lokal oder im Docker-Netzwerk)
- **Ca. 2–20 GB RAM** je nach LLM (qwen2.5:32b ~20 GB, qwen2.5:14b ~9 GB, llama3.2:3b ~2 GB)

## Schnellstart

### 1. Ollama installieren

**Lokal (macOS / Linux):**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Oder via Docker:**

```bash
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
```

### 2. Ein Modell laden

```bash
# Empfohlen (gut für RAG)
ollama pull qwen2.5:14b

# Leichter (für schwächere Hardware)
ollama pull llama3.2:3b

# Embedding-Modell (wird für die Dokumenten-Indizierung benötigt)
ollama pull nomic-embed-text
```

### 3. Boltyn einrichten

```bash
# Repository klonen
git clone https://github.com/dein-user/boltyn.git
cd boltyn

# Python-Umgebung
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Konfiguration
cp .env.example .env
# Passe .env an (z.B. OLLAMA_MODEL)
```

### 4. Starten

```bash
python server.py
```

Öffne **http://localhost:8000/admin** im Browser.

### 5. Dokumente indizieren

Lege PDFs in `docs/` ab und klicke im Admin-UI auf **"Jetzt lernen"**.  
Oder lade PDFs direkt per Drag & Drop im UI hoch.

## Docker-Komplettsetup

```bash
# Startet Ollama + Boltyn
docker-compose up -d

# Ein Modell in Ollama laden
docker exec -it boltyn-ollama ollama pull qwen2.5:14b
docker exec -it boltyn-ollama ollama pull nomic-embed-text

# UI öffnen
open http://localhost:8000/admin
```

## Konfiguration (.env)

| Variable | Default | Beschreibung |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama-Server-URL |
| `OLLAMA_MODEL` | `qwen2.5:32b` | Aktives LLM |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding-Modell |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB-Speicherort |
| `SERVER_HOST` | `0.0.0.0` | Server-Host |
| `SERVER_PORT` | `8000` | Server-Port |

## API Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| `GET` | `/api/status` | Server-Status |
| `POST` | `/api/chat` | Frage stellen |
| `GET` | `/api/models` | Verfügbare Modelle |
| `POST` | `/api/models/switch` | Modell wechseln |
| `GET` | `/api/config` | Aktuelle Konfiguration |
| `GET` | `/api/documents` | Indizierte Dokumente |
| `POST` | `/api/documents/toggle` | Dokument aktivieren/deaktivieren |
| `POST` | `/api/documents/delete` | Dokument löschen |
| `POST` | `/api/ingest` | Indizierung starten |
| `POST` | `/api/reindex` | DB neu aufbauen |
| `POST` | `/api/upload` | PDF hochladen |
| `GET` | `/api/log` | Indizierungs-Log |
| `GET/POST` | `/api/params` | RAG-Parameter |
| `POST` | `/api/restart` | Server neustarten |
| `GET` | `/admin` | Admin-UI |

## Projektstruktur

```
boltyn/
├── server.py         # FastAPI-Server (API + Admin-UI)
├── rag_engine.py     # RAG-Engine (ChromaDB + LLM)
├── ingest.py         # PDF-Ingest-Pipeline
├── config.py         # Konfiguration (.env)
├── static/
│   └── admin.html    # Admin-Frontend (alles in einer Datei)
├── docs/             # PDFs hier ablegen
├── chroma_db/        # Vektordatenbank (lokal)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Lizenz

MIT – siehe [LICENSE](LICENSE).
