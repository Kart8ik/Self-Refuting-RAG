import os
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset


DEFAULT_TEXT_FIELDS: Sequence[str] = (
    "text",
    "content",
    "document",
    "passage",
    "body",
    "article",
)


logger = logging.getLogger(__name__)


def _pick_text(row: Dict, text_fields: Sequence[str]) -> str:
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_text_corpus(
    dataset_name: Optional[str] = None,
    data_file: Optional[str] = None,
    split: str = "train",
    text_fields: Optional[Sequence[str]] = None,
    max_docs: int = 300,
) -> Tuple[List[str], List[Dict]]:
    """Load text documents and metadata from a Hugging Face dataset or local file."""
    if not dataset_name and not data_file:
        raise ValueError("Either dataset_name or data_file must be provided")

    fields = tuple(text_fields) if text_fields else DEFAULT_TEXT_FIELDS

    if dataset_name:
        logger.info("Loading Hugging Face dataset %s split=%s", dataset_name, split)
        ds = load_dataset(dataset_name, split=split)
        source_title = dataset_name
    else:
        ext = os.path.splitext(data_file or "")[1].lower().lstrip(".")
        if ext == "txt":
            with open(data_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            docs = lines[:max_docs]
            metadata = [
                {
                    "doc_id": f"local_{i}",
                    "source_title": os.path.basename(data_file or "local.txt"),
                    "chunk_index": i,
                    "text": doc,
                }
                for i, doc in enumerate(docs)
            ]
            return docs, metadata

        if ext not in {"csv", "json", "jsonl", "parquet"}:
            raise ValueError(
                "Supported file types: .txt, .csv, .json, .jsonl, .parquet"
            )

        logger.info("Loading local dataset file %s as %s", data_file, ext)
        ds = load_dataset(ext, data_files=data_file, split="train")
        source_title = os.path.basename(data_file or "dataset")

    docs: List[str] = []
    metadata: List[Dict] = []

    for idx, row in enumerate(ds):
        if len(docs) >= max_docs:
            break
        text = _pick_text(row, fields)
        if not text:
            continue

        doc_id = str(row.get("id") or row.get("doc_id") or f"doc_{idx}")
        docs.append(text)
        metadata.append(
            {
                "doc_id": doc_id,
                "source_title": source_title,
                "chunk_index": idx,
                "text": text,
            }
        )

    if not docs:
        raise ValueError(
            "No text rows found. Set text_fields to columns that contain natural language text."
        )

    logger.info("Loaded %d documents from %s", len(docs), source_title)
    return docs, metadata


def load_text_corpus_from_multiple_datasets(
    dataset_names: Sequence[str],
    split: str = "train",
    text_fields: Optional[Sequence[str]] = None,
    max_docs: int = 300,
) -> Tuple[List[str], List[Dict]]:
    """Load and combine multiple Hugging Face datasets into one corpus."""
    if not dataset_names:
        raise ValueError("dataset_names must contain at least one dataset")

    if max_docs <= 0:
        raise ValueError("max_docs must be > 0")

    per_dataset_limit = max(1, max_docs // len(dataset_names))

    all_docs: List[str] = []
    all_metadata: List[Dict] = []

    for ds_name in dataset_names:
        remaining = max_docs - len(all_docs)
        if remaining <= 0:
            break

        logger.info("Loading component dataset %s", ds_name)
        docs, metadata = load_text_corpus(
            dataset_name=ds_name,
            split=split,
            text_fields=text_fields,
            max_docs=min(per_dataset_limit, remaining),
        )
        all_docs.extend(docs)
        all_metadata.extend(metadata)

    if not all_docs:
        raise ValueError("No documents loaded from the provided datasets")

    return all_docs, all_metadata
