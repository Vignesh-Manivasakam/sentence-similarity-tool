"""
cache_manager.py — Simplified global cache manager for ChromaDB migration.

In the FAISS version this class maintained an in-memory map of
  { file_hash → path_to_.faiss_file }
and a merged in-memory LLM result cache, requiring ~150 lines of
manual file scanning, path management, and a threading-unsafe
read-modify-write JSON pattern.

With ChromaDB:
- Embedding cross-user reuse is handled by deterministic collection
  naming (ais_<hash>) — no manual map needed.
- LLM result caching retains the same interface but with a
  threading.Lock to fix the concurrent-write race condition.
"""

from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)


class GlobalCacheManager:
    """
    Manages the in-memory LLM result cache across user sessions.

    Embedding cache management has been removed — ChromaDB handles
    that transparently via persistent named collections.
    """

    def __init__(self) -> None:
        self._llm_cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_all_llm_caches()

    # ------------------------------------------------------------------
    # Embedding cache — stub kept for call-site compatibility in
    # pipeline.py / core.py. ChromaDB makes these no-ops.
    # ------------------------------------------------------------------

    def get_embedding_index_path(self, file_hash: str) -> str | None:  # noqa: ARG002
        """No-op — ChromaDB manages collection lookup internally."""
        return None

    def register_embedding(self, file_hash: str, index_path: str) -> None:  # noqa: ARG002
        """No-op — ChromaDB manages collection persistence internally."""

    # ------------------------------------------------------------------
    # LLM result cache (unchanged interface, fixed thread-safety)
    # ------------------------------------------------------------------

    def get_llm_result(self, prompt_hash: str) -> dict | None:
        with self._lock:
            return self._llm_cache.get(prompt_hash)

    def update_llm_result(self, prompt_hash: str, result: dict) -> None:
        with self._lock:
            self._llm_cache[prompt_hash] = result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_llm_caches(self) -> None:
        """Scan all user cache directories and merge LLM results into memory."""
        from am_ais_assist.config import CACHE_DIR

        if not os.path.exists(CACHE_DIR):
            logger.info("Cache directory does not exist yet — starting fresh.")
            return

        count = 0
        try:
            for user_folder in os.listdir(CACHE_DIR):
                if not user_folder.startswith("user_"):
                    continue
                # Log in the same format used throughout the app so users are
                # easily traceable in log files: "User identified: <id> (session: <id>)"
                session_id = user_folder[len("user_") :]
                logger.info("User identified: %s (session: %s)", session_id, session_id)
                llm_file = os.path.join(CACHE_DIR, user_folder, "llm_results_cache.json")
                if not os.path.exists(llm_file):
                    continue
                try:
                    with open(llm_file, encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        self._llm_cache.update(data)
                        count += len(data)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not load LLM cache from %s: %s", llm_file, exc)

            logger.info("Global LLM cache loaded: %d entries.", count)

        except OSError as exc:
            logger.error("Error scanning cache directory: %s", exc)
