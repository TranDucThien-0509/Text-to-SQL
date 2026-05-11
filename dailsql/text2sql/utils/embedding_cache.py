"""
EmbeddingCache – persistent caching for sentence embeddings.

Avoids re-encoding large training corpora across pipeline runs.
Supports both question embeddings and SQL skeleton embeddings.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Disk-backed cache for torch.Tensor embedding matrices.

    Cache files are keyed by a content hash of the texts list,
    so they automatically invalidate when the source data changes.

    Args:
        cache_dir: Directory to store cache files.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, texts: List[str]) -> Optional[torch.Tensor]:
        """
        Return cached embeddings if the cache file exists *and* the
        content hash matches.  Returns None on miss or mismatch.
        """
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                record = pickle.load(fh)
            if record.get("hash") != self._hash(texts):
                logger.debug("[EmbeddingCache] Cache stale for key='%s', rebuilding.", key)
                return None
            logger.info("[EmbeddingCache] Cache hit for key='%s'.", key)
            return record["tensor"]
        except Exception as exc:
            logger.warning("[EmbeddingCache] Failed to load cache for '%s': %s", key, exc)
            return None

    def put(self, key: str, texts: List[str], tensor: torch.Tensor) -> None:
        """Persist embeddings to disk."""
        path = self._path_for(key)
        record = {"hash": self._hash(texts), "tensor": tensor}
        try:
            with open(path, "wb") as fh:
                pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("[EmbeddingCache] Saved cache for key='%s' -> %s", key, path)
        except Exception as exc:
            logger.warning("[EmbeddingCache] Failed to save cache for '%s': %s", key, exc)

    def invalidate(self, key: str) -> None:
        path = self._path_for(key)
        if path.exists():
            path.unlink()
            logger.info("[EmbeddingCache] Invalidated cache for key='%s'.", key)

    # ── Helpers ──────────────────────────────────────────────

    def _path_for(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._cache_dir / f"{safe_key}.pkl"

    @staticmethod
    def _hash(texts: List[str]) -> str:
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8"))
        return h.hexdigest()