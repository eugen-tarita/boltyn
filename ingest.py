"""
Ingest-Pipeline: Lädt PDFs (von SharePoint oder lokal) und indiziert sie in ChromaDB.

Zwei Modi:
  1. Lokal: PDFs aus ./docs/ lesen
  2. SharePoint: PDFs von SharePoint-Listen abholen (optional, erfordert SP-Zugang)
"""
from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline as LlamaIngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings

logger = logging.getLogger("ingest")

# Globaler Ingest-Status (für /api/status)
last_ingest_time: Optional[str] = None
last_ingest_count: int = 0
ingest_running: bool = False
ingest_log: list[str] = []


def _log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    ingest_log.append(entry)
    if len(ingest_log) > 200:
        ingest_log.pop(0)
    logger.info(msg)


def _get_embed_model():
    """Try to get an embedding model – first Ollama, then fallback."""
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
        em = OllamaEmbedding(
            model_name=settings.embed_model,
            base_url=settings.ollama_base_url,
        )
        # Quick test
        em.get_text_embedding("test")
        _log(f"✅ Embedding-Modell '{settings.embed_model}' via Ollama OK")
        return em
    except Exception as e:
        _log(f"⚠️ Ollama-Embedding nicht verfügbar: {e}")
        _log("💡 Stelle sicher, dass Ollama läuft und das Modell '{}' vorhanden ist.".format(
            settings.embed_model))
        return None


def ingest_local_docs() -> dict:
    """
    Liest alle PDFs aus docs/ und indiziert sie in ChromaDB.
    Überspringt bereits indizierte Dateien (anhand des Dateinamens im Metadatum).
    """
    global last_ingest_time, last_ingest_count, ingest_running

    if ingest_running:
        return {"status": "error", "message": "Ingest läuft bereits"}

    ingest_running = True
    _log("🔍 Starte lokalen Ingest …")

    docs_dir = Path(settings.local_docs_dir)
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True)
        _log(f"📁 docs/ Verzeichnis angelegt: {docs_dir}")

    pdf_files = sorted(docs_dir.glob("*.pdf")) + sorted(docs_dir.glob("*.PDF"))
    if not pdf_files:
        _log("📭 Keine PDFs in docs/ gefunden.")
        ingest_running = False
        return {"status": "ok", "message": "Keine neuen PDFs", "count": 0}

    _log(f"📄 Gefunden: {len(pdf_files)} PDF(s)")

    try:
        # ChromaDB – immer mit cosine distance (für brauchbare Scores)
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        try:
            collection = client.get_collection(settings.collection_name)
        except Exception:
            collection = client.create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # Embedding – benötigt fürs Indizieren zwingend
        embed_model = _get_embed_model()
        if embed_model is None:
            _log("❌ Kein Embedding-Modell verfügbar – Ingest abgebrochen")
            ingest_running = False
            return {
                "status": "error",
                "message": "Kein Embedding-Modell. Stelle sicher, dass Ollama läuft.",
            }

        # Bereits bekannte Dateien abfragen
        existing = set()
        try:
            all_data = collection.get(include=["metadatas"])
            if all_data and all_data["metadatas"]:
                for m in all_data["metadatas"]:
                    fn = m.get("file_name", "")
                    if fn:
                        existing.add(fn)
        except Exception:
            pass

        new_count = 0
        skipped_count = 0
        errors = []

        for pdf_path in pdf_files:
            fname = pdf_path.name
            if fname in existing:
                _log(f"⏭️  Bereits indiziert: {fname}")
                skipped_count += 1
                continue

            try:
                _log(f"📖 Lese: {fname}")
                documents = SimpleDirectoryReader(
                    input_files=[str(pdf_path)],
                ).load_data()

                # Metadaten anreichern
                for doc in documents:
                    doc.metadata["file_name"] = fname
                    doc.metadata["source"] = f"local:{fname}"
                    doc.metadata["ingested_at"] = datetime.now().isoformat()

                # Pipeline: Chunken + Embedding + Speichern
                pipeline = LlamaIngestionPipeline(
                    transformations=[
                        SentenceSplitter(chunk_size=512, chunk_overlap=50),
                        embed_model,
                    ],
                    vector_store=vector_store,
                )

                pipeline.run(documents=documents)
                new_count += 1
                _log(f"✅ Indiziert: {fname}")

            except Exception as e:
                msg = f"❌ Fehler bei {fname}: {e}"
                _log(msg)
                errors.append(msg)

        last_ingest_time = datetime.now().isoformat()
        last_ingest_count = new_count

        result = {
            "status": "ok",
            "new": new_count,
            "skipped": skipped_count,
            "errors": len(errors),
            "total_in_db": collection.count(),
        }
        _log(f"✅ Ingest abgeschlossen: {new_count} neu, {skipped_count} übersprungen, {len(errors)} Fehler")
        ingest_running = False
        return result

    except Exception as e:
        _log(f"🔥 Ingest-Fehler: {e}")
        ingest_running = False
        return {"status": "error", "message": str(e)}


def ingest_from_sharepoint() -> dict:
    """
    Holt PDFs von SharePoint-Listen und indiziert sie.
    (Platzhalter – SharePoint-Integration folgt später)
    """
    _log("🌐 SharePoint-Ingest noch nicht implementiert – verwende zunächst lokalen Ingest.")
    _log("💡 Lege PDFs einfach in docs/ ab und rufe /api/ingest auf.")
    return {"status": "info", "message": "SharePoint-Ingest kommt in Phase 2 – nutze lokalen Ingest"}
