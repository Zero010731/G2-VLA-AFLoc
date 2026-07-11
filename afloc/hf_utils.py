"""Helpers for loading HuggingFace models in offline AFLoc runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def resolve_bert_type(default_bert_type: str) -> str:
    """Allow server runs to override checkpoint-embedded BERT paths."""
    return os.environ.get("AFLOC_BERT_TYPE", default_bert_type)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def hf_from_pretrained_kwargs(model_name_or_path: str | Path) -> dict[str, Any]:
    """Return safe kwargs for transformers.from_pretrained.

    A local path or AFLOC_HF_LOCAL_FILES_ONLY=1 disables remote HuggingFace calls.
    """
    path = Path(str(model_name_or_path))
    if path.exists() or _env_true("AFLOC_HF_LOCAL_FILES_ONLY"):
        return {"local_files_only": True}
    return {}
