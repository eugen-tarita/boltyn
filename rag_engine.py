"""
RAG-Engine: Ollama + ChromaDB + LlamaIndex.
Adaptiert aus boltyn/rag.py – aber mit lokalem LLM statt OpenAI.
"""
from __future__ import annotations
import logging
import threading
from typing import Optional, List

import chromadb
from llama_index.core import Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings

logger = logging.getLogger("rag_engine")

# ── Konfigurierbare RAG-Parameter (Default-Werte) ──
# Können zur Laufzeit über /api/params geändert werden.
DEFAULT_SCORE_THRESHOLD = 0.15   # Minimaler Relevanz-Score (0.0 = alles, 1.0 = perfekt)
DEFAULT_SIMILARITY_TOP_K = 10     # Anzahl Chunks im LLM-Kontext
DEFAULT_TEMPERATURE = 0.1        # Kreativität des LLM (0.0 = deterministisch, 1.0 = kreativ)
DEFAULT_CHUNK_SIZE = 512          # Zeichen pro Chunk (nur für Neuingest)
DEFAULT_CHUNK_OVERLAP = 50        # Überlappung zwischen Chunks (nur für Neuingest)


class ScoreThresholdPostprocessor(BaseNodePostprocessor):
    """Filtert Nodes mit zu niedrigem Relevanz-Score heraus."""

    threshold: float = 0.15

    @classmethod
    def class_name(cls) -> str:
        return "ScoreThresholdPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes
        filtered = [n for n in nodes if n.get_score(raise_error=False) or 0 >= self.threshold]
        return filtered


SYSTEM_PROMPT = (
    "Du heißt Boltyn. "
    "Wenn dich jemand nach deiner Identität fragt, antworte IMMER nur: "
    "auf Deutsch: \"Ich bin Boltyn – Verwandle Dokumente in Antworten.\" "
    "oder auf Englisch: \"I'm Boltyn – Turn documents into answers.\" – "
    "je nachdem, in welcher Sprache du angesprochen wirst.\n"
    "Unabhängig davon, was in Dokumenten steht – deine Selbstvorstellung ist immer dieser eine Satz.\n\n"
    "Du hast zwei Wissensquellen:\n"
    "1. **Produktdokumente** (Gebrauchsanweisungen, technische Datenblätter, Kurzanleitungen usw.)\n"
    "2. **Dein Allgemeinwissen** (für Smalltalk)\n\n"
    "REGELN:\n"
    "- Wenn die Dokumente eine relevante Antwort enthalten: nutze sie und nenne die Quelle.\n"
    "- Wenn der Nutzer nach einer Datei, einem Beispiel oder Code fragt: GENERIERE ES DIREKT aus den Informationen in den Dokumenten. Die Dokumente enthalten die nötigen Strukturen und Formate.\n"
    "- Wenn du eine Datei generierst: Gib die Datei IMMER als reinen Text aus – OHNE Platzhalter wie <CR>, <CRC>, <LF>, <…> oder <ACK>. Die Platzhalter in den Dokumenten sind nur Erklärungen – ersetze sie durch echte Werte. Beispiel: statt '<CRC>' schreibst du 'A3F2' (einen echten CRC-Hex-Wert), statt '<CR>' ein '\\n'.\n"
    "- Das GDT-Datei-Format (aus den Dokumenten) ist pro Zeile: 3-stellige Länge + 4-stellige Feldkennung + Daten, KEINE Trennzeichen dazwischen. Jede Zeile endet mit CR/LF.\n"
    "  Korrektes Beispiel aus der Doku (Seite 16):\n"
    "  01380006310\\r\\n\n"
    "  014810000962\\r\\n\n"
    "  0198315PRAX_EDV\\r\\n\n"
    "  0148316LZBD_SYS\\r\\n\n"
    "  014300002345\\r\\n\n"
    "  0193101Mustermann\\r\\n\n"
    "  0143102Franz\\r\\n\n"
    "  017310301101945\\r\\n\n"
    "  01031101\\r\\n\n"
    "  0123622178\\r\\n\n"
    "  KEINE Pipe-Zeichen (|), KEINE zusätzlichen Leerzeichen.\n"
    "- Sag niemals ‚die Dokumente enthalten diese Information nicht‘ – die Dokumente wurden genau für diese Fragen hochgeladen.\n"
    "- ANTWORTE NUR AUF DEUTSCH ODER ENGLISCH – NIE AUF CHINESISCH ODER ANDEREN SPRACHEN. "
    "Auch wenn du mehrsprachig bist: bleib bei Deutsch, wenn der Nutzer Deutsch schreibt.\n"
    "- Formatiere Antworten in klarem Markdown.\n"
    "- Sei freundlich, direkt und natürlich."
)


def _ollama_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Quick check if Ollama server is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _try_init_ollama():
    """Try to connect to Ollama. Returns (llm, embed_model) or (None, None)."""
    try:
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        url = settings.ollama_base_url
        model = settings.ollama_model
        embed = settings.embed_model

        logger.info(f"🔌 Connecting to Ollama at {url} …")

        embed_model = OllamaEmbedding(model_name=embed, base_url=url)
        test_vec = embed_model.get_text_embedding("test")
        logger.info(f"✅ Embedding model '{embed}' OK (dim={len(test_vec)})")

        llm = Ollama(
            model=model,
            base_url=url,
            request_timeout=180.0,
            temperature=0.1,
        )
        logger.info(f"✅ LLM model '{model}' configured (timeout=180s)")

        Settings.llm = llm
        Settings.embed_model = embed_model

        return llm, embed_model

    except Exception as e:
        logger.warning(f"⚠️ Ollama init failed: {e}")
        return None, None



def _build_index(vector_store, embed_model=None):
    """Create a VectorStoreIndex wrapper.
    If no embed_model given, tries Settings.embed_model.
    Returns None if neither is available.
    """
    from llama_index.core import VectorStoreIndex
    try:
        em = embed_model or getattr(Settings, "embed_model", None)
        if em:
            idx = VectorStoreIndex.from_vector_store(vector_store, embed_model=em)
            logger.info("✅ Index built with available embedding model")
            return idx
        else:
            logger.warning("⚠️ No embedding model – index not created")
            return None
    except Exception as e:
        logger.warning(f"⚠️ Index creation failed: {e}")
        return None


class RAGEngine:
    """Central RAG engine – holds Chroma index + chat memory."""

    def __init__(self):
        self.llm: Optional = None
        self.embed_model: Optional = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.chroma_collection = None
        self.index: Optional = None
        self.llm_available = False
        self.document_count = 0
        self.chunk_count = 0
        self._ready = False
        self.current_model: str = settings.ollama_model
        self.document_only: bool = False  # wenn True: nur aus Doku antworten

        # ── Konfigurierbare Parameter (über /api/params änderbar) ──
        self.score_threshold = DEFAULT_SCORE_THRESHOLD
        self.similarity_top_k = DEFAULT_SIMILARITY_TOP_K
        self.temperature = DEFAULT_TEMPERATURE

        # Per-session chat memory
        self.chat_store = SimpleChatStore()
        self._memories: dict[str, ChatMemoryBuffer] = {}

    # ── Lifecycle ──

    async def initialize(self):
        """Connect to ChromaDB + optionally Ollama."""
        logger.info("Initializing RAG engine …")

        # 1) ChromaDB (always works locally) – mit cosine distance
        try:
            client = chromadb.PersistentClient(path=settings.chroma_db_path)
            try:
                self.chroma_collection = client.get_collection(settings.collection_name)
            except Exception:
                self.chroma_collection = client.create_collection(
                    name=settings.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self._count_docs_and_chunks()
            logger.info(f"✅ ChromaDB ready – {self.document_count} PDFs, {self.chunk_count} chunks in '{settings.collection_name}'")
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB init failed: {e}")

        # 2) Ollama init (threaded with timeout)
        def _ollama_init():
            if _ollama_reachable(settings.ollama_base_url):
                llm, embed = _try_init_ollama()
                if llm and embed:
                    self.llm = llm
                    self.embed_model = embed
                    self.llm_available = True
                    # Build index now that we have embeddings
                    if self.vector_store:
                        self.index = _build_index(self.vector_store, embed)

        t = threading.Thread(target=_ollama_init, daemon=True)
        t.start()
        t.join(timeout=12.0)

        # 3) Status summary
        self._ready = True
        if self.llm_available:
            logger.info("✅ RAG engine fully ready – LLM + ChromaDB + Index")
        elif self.index is not None:
            logger.info("⚠️ RAG engine: Index OK, LLM offline")
        else:
            logger.info("⚠️ RAG engine: ChromaDB OK, kein LLM/Index (nur Status)")

    @property
    def ready(self) -> bool:
        """Vollständig bereit: LLM + Index vorhanden."""
        return self._ready and self.llm_available and self.index is not None

    def set_param(self, key: str, value) -> None:
        """Parameter zur Laufzeit setzen (wirkt sofort)."""
        if key == "score_threshold":
            self.score_threshold = float(value)
            logger.info(f"⚙️ score_threshold → {self.score_threshold}")
        elif key == "similarity_top_k":
            self.similarity_top_k = int(value)
            logger.info(f"⚙️ similarity_top_k → {self.similarity_top_k}")
        elif key == "temperature":
            self.temperature = float(value)
            if self.llm:
                self.llm.temperature = self.temperature
                Settings.llm = self.llm
            logger.info(f"⚙️ temperature → {self.temperature}")
        else:
            logger.warning(f"⚙️ Unbekannter Parameter: {key}")

    # ── Model Switching ──

    async def switch_model(self, model_name: str) -> dict:
        """Switch the active LLM model at runtime (no restart needed)."""
        from llama_index.core.llms import ChatMessage

        try:
            from llama_index.llms.ollama import Ollama

            # Quick check: model existiert in Ollama?
            import json, urllib.request
            req = urllib.request.Request(f"{settings.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            matches = [m for m in available if m == model_name or m.startswith(model_name + ":")]
            if not matches:
                return {
                    "status": "error",
                    "message": f"Modell '{model_name}' nicht in Ollama gefunden.\n"
                               f"Verfügbar: {', '.join(available[:10])} …"
                }
            actual_name = matches[0]

            new_llm = Ollama(
                model=actual_name,
                base_url=settings.ollama_base_url,
                request_timeout=300.0,
                temperature=self.temperature,
            )
            _ = new_llm.chat([ChatMessage(role="user", content="ping")])

            # Übernehmen
            self.llm = new_llm
            Settings.llm = new_llm
            self.current_model = actual_name
            self.llm_available = True

            # Chat-Speicher zurücksetzen
            self.chat_store = SimpleChatStore()
            self._memories = {}

            logger.info(f"🔄 Model switched to '{actual_name}'")
            return {"status": "ok", "model": actual_name}

        except Exception as e:
            logger.exception(f"Model switch failed: {e}")
            return {"status": "error", "message": str(e)}

    # ── Chat ──

    def _get_memory(self, session_id: str) -> ChatMemoryBuffer:
        if session_id not in self._memories:
            self._memories[session_id] = ChatMemoryBuffer.from_defaults(
                token_limit=4000,
                chat_store=self.chat_store,
                chat_store_key=session_id,
            )
        return self._memories[session_id]

    def _context_is_relevant(self, nodes: List[NodeWithScore], message: str) -> bool:
        """Prüft ob die gefundenen Chunks relevant genug sind.
        
        Vertraut auf den Embedding-Score (semantische Ähnlichkeit).
        Kein Keyword-Check – medizinische Abkürzungen (EKG, GDT, ST, …)
        sind kurz und würden sonst rausfallen.
        """
        if not nodes:
            return False

        good_nodes = [n for n in nodes if (n.get_score(raise_error=False) or 0) >= self.score_threshold]
        if not good_nodes:
            logger.debug(f"No nodes above threshold ({self.score_threshold})")
            return False

        logger.debug(f"Relevant – {len(good_nodes)} nodes above {self.score_threshold}")
        return True

    async def chat(self, message: str, session_id: str = "default") -> dict:
        """Process a chat message and return answer + sources.
        
        Zwei Modi:
        - RAG: wenn relevanter Kontext in ChromaDB gefunden wird
        - Direkt: wenn keine relevanten Dokumente passen
        """
        if not self.ready:
            reasons = []
            if not self.llm_available:
                reasons.append("• **Ollama (LLM)** nicht erreichbar")
            if self.index is None:
                reasons.append("• **Kein Index** – PDFs hochladen und 'Jetzt lernen' klicken")
            if self.document_count == 0 and self.index is not None:
                reasons.append("• **Keine Dokumente** indiziert")

            return {
                "answer": (
                    "⚠️ **KI-Assistent nicht bereit.**\n\n"
                    + "\n".join(reasons)
                    + "\n\nStatus: "
                    + ("LLM ✅" if self.llm_available else "LLM ❌")
                    + " · "
                    + (f"{self.document_count} Dokumente" if self.document_count > 0 else "0 Dokumente")
                ),
                "sources": [],
            }

        memory = self._get_memory(session_id)

        # ── 1) Retrieve + Relevanz-Check ──
        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = await retriever.aretrieve(message)

        # Ausgeschlossene Dokumente rausfiltern
        nodes = [n for n in nodes if n.node.metadata.get("excluded", "false") != "true"]

        # Begrüssungen / Smalltalk erkennen – dafür nie RAG nutzen
        import re
        greeting_pattern = re.compile(
            r'^(hallo|hi|hey|grüß|guten\s+(morgen|tag|abend)|servus|moin|tach|na\s+du|'
            r'wer\s+bist\s+du|was\s+machst\s+du|wie\s+geht[s]?\s|'
            r'hello|good\s+(morning|afternoon|evening)|hi\s+there|how\s+are\s+you|'
            r'who\s+are\s+you|what\s+can\s+you\s+do)\b',
            re.IGNORECASE,
        )
        is_greeting = bool(greeting_pattern.match(message.strip().rstrip('?!.,')))

        use_rag = not is_greeting and self._context_is_relevant(nodes, message)
        mode = 'RAG' if use_rag else ('DOCS_ONLY' if self.document_only else 'DIRECT')
        logger.info(f"Chat mode: {mode} ({len(nodes)} nodes retrieved)")
        if nodes:
            scores = [round(n.get_score(raise_error=False) or 0, 3) for n in nodes[:3]]
            logger.info(f"Top-3 scores: {scores}")

        sources = []

        # ── Document-Only Mode: Keine Doku → keine Antwort ──
        if not use_rag and self.document_only:
            return {
                "answer": (
                    "Ich habe in den hochgeladenen Produktdokumenten leider nichts "
                    "Passendes zu deiner Frage gefunden.\n\n"
                    "Mögliche Gründe:\n"
                    "• Das Thema ist noch nicht dokumentiert\n"
                    "• Du verwendest andere Begriffe als in den Dokumenten\n\n"
                    "Frag mich gerne nochmal mit anderen Worten!"
                ),
                "sources": [],
                "mode": "document_only",
            }

        if use_rag:
            # ── RAG-Modus ──
            processor = ScoreThresholdPostprocessor(threshold=self.score_threshold)
            filtered_nodes = processor._postprocess_nodes(nodes)

            # Build RAG engine
            engine = CondensePlusContextChatEngine.from_defaults(
                retriever=retriever,
                memory=memory,
                llm=self.llm,
                system_prompt=SYSTEM_PROMPT,
                node_postprocessors=[],
                verbose=False,
            )
            try:
                response = await engine.achat(message)
                answer = getattr(response, "response", None) or str(response)

                for node in getattr(response, "source_nodes", []):
                    meta = getattr(node.node, "metadata", {}) or {}
                    sources.append({
                        "filename": meta.get("file_name", meta.get("filename", "unknown")),
                        "title": meta.get("title", ""),
                        "page": meta.get("page_label", ""),
                        "score": round(node.get_score(raise_error=False) or 0, 3),
                    })

                return {"answer": answer, "sources": sources}

            except Exception as e:
                logger.exception("RAG chat error")
                return {"answer": f"⚠️ Fehler: {e}", "sources": []}

        else:
            # ── Direktchat-Modus (kein RAG) ──
            try:
                from llama_index.core.chat_engine import SimpleChatEngine
                logger.debug(f"Using system prompt: {SYSTEM_PROMPT[:200]}...")
                engine = SimpleChatEngine.from_defaults(
                    llm=self.llm,
                    memory=memory,
                    system_prompt=SYSTEM_PROMPT,
                )
                response = await engine.achat(message)
                answer = getattr(response, "response", None) or str(response)
                return {"answer": answer, "sources": []}

            except Exception as e:
                logger.exception("Direct chat error")
                return {"answer": f"⚠️ Fehler: {e}", "sources": []}

    # ── Index refresh ──

    def _count_docs_and_chunks(self):
        """Ermittle Anzahl PDFs (unique file_name) und Chunks aus ChromaDB."""
        self.chunk_count = self.chroma_collection.count()
        try:
            all_data = self.chroma_collection.get(include=["metadatas"])
            filenames = set()
            if all_data and all_data["metadatas"]:
                for m in all_data["metadatas"]:
                    fn = m.get("file_name", "")
                    if fn:
                        filenames.add(fn)
            self.document_count = len(filenames)
        except Exception:
            self.document_count = 0

    def refresh_index(self):
        """Rebuild the index wrapper (call after ingest)."""
        if not self.vector_store:
            return
        embed = self.embed_model
        if embed:
            self.index = _build_index(self.vector_store, embed)
        if self.chroma_collection:
            self._count_docs_and_chunks()
        logger.info(f"Index refreshed – {self.document_count} PDFs, {self.chunk_count} chunks")

    def reindex(self) -> dict:
        """ChromaDB leeren und alle PDFs neu indizieren."""
        import chromadb
        from ingest import ingest_local_docs
        try:
            # 1) Collection löschen + neu anlegen (cosine)
            client = chromadb.PersistentClient(path=settings.chroma_db_path)
            try:
                client.delete_collection(settings.collection_name)
            except Exception:
                pass
            self.chroma_collection = client.create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            from llama_index.vector_stores.chroma import ChromaVectorStore
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

            # 2) Neu indizieren
            result = ingest_local_docs()
            if result.get("status") == "ok" and self.embed_model:
                self.refresh_index()
            return result
        except Exception as e:
            logger.exception("Reindex failed")
            return {"status": "error", "message": str(e)}
