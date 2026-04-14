import os
import logging
import http.client as http_client
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv, dotenv_values, set_key
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import run_query
from sr_rag.config import load_config
from sr_rag.retrieval.dataset_loader import (
    load_text_corpus,
    load_text_corpus_from_multiple_datasets,
)
from sr_rag.retrieval.vector_index import VectorIndex


load_dotenv()

DEBUG_LOGS = os.environ.get("SR_RAG_DEBUG_LOGS", "1").strip().lower() not in {"0", "false", "no"}
LOG_PATH = Path(__file__).resolve().parent / "logs" / "terminal" / "backend.log"
ENV_PATH = Path(__file__).resolve().parent / ".env"
INDEX_LOCK = threading.RLock()
CURRENT_CORPUS: Dict[str, Any] = {
    "name": "built-in sample corpus",
    "source": "fallback",
    "document_count": 3,
    "chunk_count": 3,
    "files": ["built-in sample"],
}
KEY_USAGE_LOCK = threading.RLock()
KEY_LAST_USED: Dict[str, str] = {}
ACTIVE_GROQ_KEY_NAME = os.environ.get("SR_RAG_ACTIVE_GROQ_KEY_NAME", "GROQ_API_KEY")


def _is_groq_key_name(name: str) -> bool:
    if name == "GROQ_API_KEY":
        return True
    if name.startswith("GROQ_API_KEY_"):
        return True
    return name.startswith("GROQ_") and "KEY" in name


def _configure_logging() -> None:
    level = logging.DEBUG if DEBUG_LOGS else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    if DEBUG_LOGS:
        http_client.HTTPConnection.debuglevel = 1

    for logger_name in ("datasets", "huggingface_hub", "urllib3", "requests", "httpx", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(level)
        logging.getLogger(logger_name).propagate = True

    if DEBUG_LOGS:
        try:
            from datasets.utils import logging as datasets_logging

            datasets_logging.set_verbosity_debug()
        except Exception:
            pass

        try:
            from huggingface_hub.utils import logging as hf_logging

            hf_logging.set_verbosity_debug()
        except Exception:
            pass

        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_info()
        except Exception:
            pass


_configure_logging()
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SR-RAG Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):(\d+)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    overall_confidence: float
    claim_table: Optional[List[Dict[str, Any]]]
    metadata: Dict[str, Any]
    key_info: Optional[Dict[str, Any]] = None


class KeyOption(BaseModel):
    name: str
    masked_value: str
    active: bool
    last_used_at: Optional[str] = None


class KeyListResponse(BaseModel):
    active_key_name: Optional[str]
    keys: List[KeyOption]


class KeySelectRequest(BaseModel):
    name: str
    persist: bool = True


class KeySelectResponse(BaseModel):
    status: str
    active_key_name: str


class CorpusUploadRequest(BaseModel):
    filename: str
    content: str
    replace: bool = True
    chunk_size: int = 1200
    overlap: int = 150


class CorpusResponse(BaseModel):
    status: str
    corpus: Dict[str, Any]
    sample_chunks: List[str]


class LogResponse(BaseModel):
    path: str
    lines: List[str]


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _collect_groq_keys() -> Dict[str, str]:
    keys: Dict[str, str] = {}
    env_map = dict(os.environ)

    for key, value in env_map.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if _is_groq_key_name(key):
            keys[key] = value.strip()

    # Parse .env directly so repeated GROQ_API_KEY lines are all visible/selectable.
    if ENV_PATH.exists():
        try:
            with ENV_PATH.open("r", encoding="utf-8", errors="replace") as fh:
                raw_lines = fh.readlines()
        except Exception:
            raw_lines = []

        duplicate_index = 0
        for raw in raw_lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not _is_groq_key_name(key):
                continue

            value = value.strip().strip('"').strip("'")
            if not value:
                continue

            if key == "GROQ_API_KEY":
                duplicate_index += 1
                alias = f"GROQ_API_KEY_ENV_{duplicate_index}"
                keys[alias] = value
            elif key not in keys:
                keys[key] = value

    try:
        file_map = dotenv_values(str(ENV_PATH)) if ENV_PATH.exists() else {}
        for k, v in file_map.items():
            if isinstance(v, str) and v.strip() and _is_groq_key_name(k) and k not in keys:
                keys[k] = v.strip()
    except Exception:
        pass

    return dict(sorted(keys.items()))


def _set_active_groq_key(name: str, persist: bool = True) -> None:
    global ACTIVE_GROQ_KEY_NAME
    keys = _collect_groq_keys()
    if name not in keys:
        raise ValueError(f"Unknown Groq key name: {name}")

    selected_value = keys[name]
    os.environ["GROQ_API_KEY"] = selected_value
    os.environ["SR_RAG_ACTIVE_GROQ_KEY_NAME"] = name
    ACTIVE_GROQ_KEY_NAME = name

    if persist:
        try:
            if not ENV_PATH.exists():
                ENV_PATH.touch()
            set_key(str(ENV_PATH), "GROQ_API_KEY", selected_value)
            set_key(str(ENV_PATH), "SR_RAG_ACTIVE_GROQ_KEY_NAME", name)
        except Exception as exc:
            raise RuntimeError(f"Failed to persist key selection to .env: {exc}")


def _active_key_name() -> Optional[str]:
    keys = _collect_groq_keys()
    selected = os.environ.get("SR_RAG_ACTIVE_GROQ_KEY_NAME") or ACTIVE_GROQ_KEY_NAME
    if selected in keys:
        return selected

    current_secret = os.environ.get("GROQ_API_KEY", "")
    if current_secret:
        for name, value in keys.items():
            if value == current_secret:
                return name
    return None


def _record_key_usage(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    stamp = datetime.now(timezone.utc).isoformat()
    with KEY_USAGE_LOCK:
        KEY_LAST_USED[name] = stamp
    return stamp


def _fallback_corpus() -> tuple[list[str], list[dict]]:
    docs = [
        "The SR-RAG architecture was proposed by Team 20 for their GenAI course in March 2026.",
        "It uses a multi-agent system consisting of Proposer, Decomposer, Refuter, and Judge.",
        "The Refuter operates using only retrieved documents to challenge low confidence claims.",
    ]
    metadata = [
        {"doc_id": "d1", "source_title": "Project Info", "chunk_index": 0, "text": docs[0]},
        {"doc_id": "d2", "source_title": "Project Info", "chunk_index": 0, "text": docs[1]},
        {"doc_id": "d3", "source_title": "Project Info", "chunk_index": 0, "text": docs[2]},
    ]
    return docs, metadata


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""

            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + chunk_size)
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _build_index_from_uploaded_text(filename: str, content: str, chunk_size: int, overlap: int):
    chunks = _chunk_text(content, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Uploaded file did not contain readable text.")

    metadata = []
    for idx, chunk in enumerate(chunks):
        metadata.append(
            {
                "doc_id": f"{filename}_{idx}",
                "source_title": filename,
                "chunk_index": idx,
                "text": chunk,
            }
        )

    corpus = {
        "name": filename,
        "source": "upload",
        "document_count": 1,
        "chunk_count": len(chunks),
        "files": [filename],
    }
    return chunks, metadata, corpus


def _build_index_from_env() -> VectorIndex:
    index = VectorIndex()

    dataset_name = os.environ.get("E2E_DATASET_NAME")
    dataset_names_env = os.environ.get("E2E_DATASET_NAMES", "")
    dataset_names = [d.strip() for d in dataset_names_env.split(",") if d.strip()]
    dataset_file = os.environ.get("E2E_DATA_FILE")
    dataset_split = os.environ.get("E2E_DATASET_SPLIT", "train")
    max_docs = int(os.environ.get("E2E_MAX_DOCS", "300"))
    text_fields_env = os.environ.get("E2E_TEXT_FIELDS", "")
    text_fields = [f.strip() for f in text_fields_env.split(",") if f.strip()] or None

    docs, metadata = _fallback_corpus()

    if dataset_names or dataset_name or dataset_file:
        try:
            if dataset_names:
                docs, metadata = load_text_corpus_from_multiple_datasets(
                    dataset_names=dataset_names,
                    split=dataset_split,
                    text_fields=text_fields,
                    max_docs=max_docs,
                )
            else:
                docs, metadata = load_text_corpus(
                    dataset_name=dataset_name,
                    data_file=dataset_file,
                    split=dataset_split,
                    text_fields=text_fields,
                    max_docs=max_docs,
                )
        except Exception as exc:
            print(f"Dataset load failed, falling back to built-in corpus: {exc}")

    index.build(docs, metadata)
    corpus_name = "built-in sample corpus"
    corpus_source = "fallback"
    corpus_files = ["built-in sample"]

    if dataset_names:
        corpus_name = ", ".join(dataset_names)
        corpus_source = "datasets"
        corpus_files = dataset_names
    elif dataset_name:
        corpus_name = dataset_name
        corpus_source = "datasets"
        corpus_files = [dataset_name]
    elif dataset_file:
        corpus_name = Path(dataset_file).name
        corpus_source = "local-file"
        corpus_files = [dataset_file]

    CURRENT_CORPUS.update(
        {
            "name": corpus_name,
            "source": corpus_source,
            "document_count": len(docs),
            "chunk_count": len(metadata),
            "files": corpus_files,
        }
    )
    return index


_CONFIG = load_config()
_INDEX = _build_index_from_env()

initial_key_name = _active_key_name()
if initial_key_name:
    _set_active_groq_key(initial_key_name, persist=False)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/keys", response_model=KeyListResponse)
def list_keys() -> KeyListResponse:
    keys = _collect_groq_keys()
    active_name = _active_key_name()
    with KEY_USAGE_LOCK:
        usage_snapshot = dict(KEY_LAST_USED)

    key_items = [
        KeyOption(
            name=name,
            masked_value=_mask_secret(value),
            active=name == active_name,
            last_used_at=usage_snapshot.get(name),
        )
        for name, value in keys.items()
    ]
    return KeyListResponse(active_key_name=active_name, keys=key_items)


@app.post("/api/keys/select", response_model=KeySelectResponse)
def select_key(payload: KeySelectRequest) -> KeySelectResponse:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name cannot be empty")

    try:
        _set_active_groq_key(name, persist=payload.persist)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return KeySelectResponse(status="ok", active_key_name=name)


@app.get("/api/corpus")
def corpus_info() -> Dict[str, Any]:
    with INDEX_LOCK:
        return {"corpus": CURRENT_CORPUS}


@app.get("/api/logs", response_model=LogResponse)
def recent_logs(lines: int = 200) -> LogResponse:
    lines = max(1, min(lines, 1000))
    if not LOG_PATH.exists():
        return LogResponse(path=str(LOG_PATH), lines=["Log file not found yet."])

    content = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return LogResponse(path=str(LOG_PATH), lines=content[-lines:])


@app.post("/api/corpus/upload", response_model=CorpusResponse)
def upload_corpus(payload: CorpusUploadRequest) -> CorpusResponse:
    filename = payload.filename.strip() or "uploaded_document.txt"
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content cannot be empty")

    chunk_size = max(200, min(payload.chunk_size, 4000))
    overlap = max(0, min(payload.overlap, chunk_size - 1))

    docs, metadata, corpus = _build_index_from_uploaded_text(filename, content, chunk_size, overlap)

    global _INDEX, CURRENT_CORPUS
    with INDEX_LOCK:
        if payload.replace:
            _INDEX = VectorIndex()
            _INDEX.build(docs, metadata)
        else:
            _INDEX.build(docs, metadata)
            corpus["document_count"] = CURRENT_CORPUS.get("document_count", 0) + 1
            corpus["chunk_count"] = CURRENT_CORPUS.get("chunk_count", 0) + len(metadata)

        CURRENT_CORPUS = corpus

    return CorpusResponse(status="ok", corpus=corpus, sample_chunks=docs[:3])


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    with INDEX_LOCK:
        current_index = _INDEX

    active_key = _active_key_name()
    used_at = _record_key_usage(active_key)
    output = run_query(message, _CONFIG, current_index)
    return ChatResponse(
        answer=output.natural_language_answer,
        overall_confidence=output.overall_confidence,
        claim_table=output.claim_table,
        metadata=output.metadata,
        key_info={
            "active_key_name": active_key,
            "last_used_at": used_at,
        },
    )
