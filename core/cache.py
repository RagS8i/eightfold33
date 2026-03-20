"""
Response Cache — SHA-256 keyed LRU cache.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import LRUCache
from config import settings


class EvaluationCache:
    def __init__(self, maxsize: int | None = None):
        self._cache: LRUCache = LRUCache(maxsize=maxsize or settings.CACHE_MAX_SIZE)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(jd: dict, candidate: dict) -> str:
        raw = json.dumps({"jd": jd, "candidate": candidate}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, jd: dict, candidate: dict) -> Any | None:
        key = self._make_key(jd, candidate)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, jd: dict, candidate: dict, result: Any) -> None:
        key = self._make_key(jd, candidate)
        self._cache[key] = result

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "current_size": len(self._cache),
            "max_size": self._cache.maxsize,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


_cache: EvaluationCache | None = None


def get_cache() -> EvaluationCache:
    global _cache
    if _cache is None:
        _cache = EvaluationCache()
    return _cache
