#!/usr/bin/env python3
"""
BOLTYN – Produktdokumentations-Assistent

Eigenständiger FastAPI-Server mit:
  • /admin   – HTML-Admin-Seite (Status + Chat + Lernen)
  • /api/chat   – Chat für WP-Integration
  • /api/ingest – Lernen triggern
  • /api/status – Status der Vektordatenbank

Läuft parallel zu WordPress im lokalen Netz (NAS).
"""
from __future__ import annotations
import logging
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from config import settings
from rag_engine import RAGEngine
from ingest import ingest_local_docs, ingest_from_sharepoint, ingest_log, last_ingest_time, last_ingest_count, ingest_running

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("server")

# ── App ──
app = FastAPI(
    title="BOLTYN – Turn documents into answers",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)

rag = RAGEngine()

STATIC_DIR = Path(__file__).parent / "static"


# ── Lifecycle ──

@app.on_event("startup")
async def startup():
    logger.info("🚀 BOLTYN starting up …")
    await rag.initialize()
    logger.info("✅ BOLTYN ready")


# ── API-Endpunkte ──

@app.get("/api/status")
async def api_status():
    """Status der Vektordatenbank und LLM-Verbindung."""
    return {
        "llm_available": rag.llm_available,
        "ollama_model": getattr(rag, "current_model", settings.ollama_model),
        "document_count": rag.document_count,
        "chunk_count": rag.chunk_count,
        "chroma_db_path": settings.chroma_db_path,
        "last_ingest_time": last_ingest_time,
        "last_ingest_count": last_ingest_count,
        "ingest_running": ingest_running,
        "ready": rag.ready,
        "document_only": rag.document_only,
    }


@app.get("/api/config")
async def api_config_get():
    """Aktuelle Konfiguration abrufen."""
    return {
        "document_only": rag.document_only,
    }


@app.post("/api/config")
async def api_config_set(req: Request):
    """Konfiguration aktualisieren (z.B. document_only)."""
    body = await req.json()
    if "document_only" in body:
        rag.document_only = bool(body["document_only"])
        logger.info(f"⚙️ document_only = {rag.document_only}")
    return {
        "status": "ok",
        "document_only": rag.document_only,
    }


# ── RAG-Parameter (Laufzeit-Konfiguration) ──

PARAM_DEFS = {
    "score_threshold": {
        "label": "Min. Relevanz-Score",
        "description": "Chunks unter diesem Score werden ignoriert. 0.0 = alles akzeptieren, 1.0 = nur perfekte Treffer.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
    },
    "similarity_top_k": {
        "label": "Chunks im Kontext",
        "description": "Wieviele Dokument-Chunks ans LLM übergeben werden. Mehr = mehr Kontext, aber langsamer.",
        "type": "int",
        "min": 1,
        "max": 50,
        "step": 1,
        "unit": "Stück",
    },
    "temperature": {
        "label": "Kreativität (Temperature)",
        "description": "0.0 = deterministisch/faktentreu, 1.0 = kreativ/frei. Für medizinische Fragen niedrig lassen.",
        "type": "float",
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "unit": "",
    },
}

@app.get("/api/params")
async def api_params_get():
    """Alle RAG-Parameter mit Definitionen abrufen."""
    vals = {
        "score_threshold": rag.score_threshold,
        "similarity_top_k": rag.similarity_top_k,
        "temperature": rag.temperature,
    }
    return {"values": vals, "definitions": PARAM_DEFS}


@app.post("/api/params")
async def api_params_set(req: Request):
    """RAG-Parameter setzen (wirkt sofort, kein Neustart nötig)."""
    body = await req.json()
    results = {}
    for key, val in body.items():
        if key not in PARAM_DEFS:
            results[key] = "unbekannt"
            continue
        try:
            rag.set_param(key, val)
            results[key] = str(getattr(rag, key, val))
        except (ValueError, TypeError) as e:
            results[key] = f"Fehler: {e}"
    return {"status": "ok", "values": results}


@app.get("/api/models")
async def api_models():
    """List available models from Ollama with sizes."""
    try:
        import json
        import urllib.request
        req = urllib.request.Request(f"{settings.ollama_base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        model_list = []
        for m in data.get("models", []):
            size_bytes = m.get("size", 0)
            size_gb = round(size_bytes / (1024 ** 3), 1)
            model_list.append({
                "name": m["name"],
                "size_gb": size_gb,
                "size_bytes": size_bytes,
            })

        model_list.sort(key=lambda x: x["name"])
        return {
            "models": model_list,
            "current": getattr(rag, "current_model", settings.ollama_model),
        }
    except Exception as e:
        return JSONResponse(
            {"models": [], "current": settings.ollama_model, "error": str(e)},
            status_code=503,
        )


@app.post("/api/models/switch")
async def api_switch_model(req: Request):
    """Switch the active LLM model at runtime."""
    body = await req.json()
    model = body.get("model", "").strip()
    if not model:
        return JSONResponse({"status": "error", "message": "Kein Modell angegeben"})
    result = await rag.switch_model(model)
    return JSONResponse(result)


@app.post("/api/chat")
async def api_chat(req: Request):
    """Chat-Anfrage (genutzt von Admin + WordPress)."""
    body = await req.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not message:
        return JSONResponse({"answer": "⚠️ Bitte gib eine Nachricht ein.", "sources": []})

    result = await rag.chat(message, session_id)
    return JSONResponse(result)


@app.post("/api/ingest")
async def api_ingest():
    """Dokumente lernen / neu indizieren."""
    result = ingest_local_docs()
    if result.get("status") == "ok" and rag.embed_model:
        rag.refresh_index()
    return JSONResponse(result)


@app.post("/api/ingest/sharepoint")
async def api_ingest_sharepoint():
    """(Vorbereitet) SharePoint-Ingest."""
    result = ingest_from_sharepoint()
    return JSONResponse(result)


@app.post("/api/reindex")
async def api_reindex():
    """ChromaDB leeren + alle PDFs neu indizieren."""
    result = rag.reindex()
    return JSONResponse(result)


@app.post("/api/restart")
async def api_restart(bg: BackgroundTasks):
    """Server neustarten (nach Response)."""
    bg.add_task(_do_restart)
    return JSONResponse({"status": "ok", "message": "Server wird neu gestartet …"})


def _do_restart():
    import time, os, sys
    time.sleep(0.5)
    os.execv(sys.executable, [sys.executable] + sys.argv)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    Einzelnes PDF hochladen und sofort indizieren.
    Erlaubt schnelles Testen ohne docs/-Verzeichnis.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"status": "error", "message": "Nur PDF-Dateien erlaubt"}, status_code=400)

    docs_dir = Path(settings.local_docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / file.filename

    content = await file.read()
    dest.write_bytes(content)
    logger.info(f"📄 Upload: {file.filename} ({len(content)} bytes)")

    # Sofort indizieren
    result = ingest_local_docs()
    if result.get("status") == "ok" and rag.embed_model:
        rag.refresh_index()

    return JSONResponse({
        "status": "ok",
        "filename": file.filename,
        "ingest_result": result,
    })


@app.get("/api/log")
async def api_log():
    """Letzte Ingest-Log-Einträge."""
    return {"log": list(ingest_log)}


# ── Dokumenten-Management ──

@app.get("/api/documents")
async def api_documents():
    """Alle indizierten Dokumente mit Status auflisten."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_or_create_collection(settings.collection_name)
        all_data = collection.get(include=["metadatas"])
    except Exception as e:
        return JSONResponse({"documents": [], "error": str(e)})

    docs: dict[str, dict] = {}
    for meta in (all_data.get("metadatas") or []):
        fn = meta.get("file_name") or "unknown"
        if fn not in docs:
            docs[fn] = {
                "filename": fn,
                "file_size": meta.get("file_size", 0),
                "page_label": meta.get("page_label", ""),
                "source": meta.get("source", ""),
                "ingested_at": meta.get("ingested_at", ""),
                "excluded": meta.get("excluded", "false") == "true",
                "chunks": 0,
            }
        docs[fn]["chunks"] += 1

    return {"documents": sorted(docs.values(), key=lambda d: d["filename"])}


@app.post("/api/documents/toggle")
async def api_document_toggle(req: Request):
    """Dokument von der Suche aus-/einschließen."""
    body = await req.json()
    filename = body.get("filename", "").strip()
    excluded = body.get("excluded", True)

    if not filename:
        return JSONResponse({"status": "error", "message": "Kein Dateiname"})

    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_or_create_collection(settings.collection_name)

        # Alle Chunks dieses Dokuments finden
        result = collection.get(where={"file_name": filename}, include=["metadatas"])
        ids = result.get("ids", [])
        if not ids:
            return JSONResponse({"status": "error", "message": f"'{filename}' nicht im Index"})

        excluded_str = "true" if excluded else "false"
        collection.update(ids=ids, metadatas=[{"excluded": excluded_str}] * len(ids))
        logger.info(f"📄 '{filename}' → excluded={excluded} ({len(ids)} chunks)")

        return {"status": "ok", "filename": filename, "excluded": excluded, "chunks_updated": len(ids)}
    except Exception as e:
        logger.exception("Toggle failed")
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/documents/delete")
async def api_document_delete(req: Request):
    """Dokument komplett aus dem Index löschen."""
    body = await req.json()
    filename = body.get("filename", "").strip()

    if not filename:
        return JSONResponse({"status": "error", "message": "Kein Dateiname"})

    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_or_create_collection(settings.collection_name)

        result = collection.get(where={"file_name": filename}, include=["metadatas"])
        ids = result.get("ids", [])
        if not ids:
            return JSONResponse({"status": "error", "message": f"'{filename}' nicht im Index"})

        collection.delete(ids=ids)
        rag.refresh_index()
        logger.info(f"🗑️ '{filename}' gelöscht ({len(ids)} chunks)")

        return {"status": "ok", "filename": filename, "chunks_deleted": len(ids)}
    except Exception as e:
        logger.exception("Delete failed")
        return JSONResponse({"status": "error", "message": str(e)})


# ── Admin-Frontend ──

@app.get("/admin")
async def admin_page():
    """All-in-One Admin-Seite (Status, Chat, Lernen)."""
    html_path = STATIC_DIR / "admin.html"
    if not html_path.exists():
        return HTMLResponse("admin.html not found", status_code=404)
    return FileResponse(str(html_path))


@app.get("/")
async def root():
    """Weiterleitung zum Admin."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/admin")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
        reload=False,
    )
