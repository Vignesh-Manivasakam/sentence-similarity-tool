"""
feedback_store.py — Per-user and global feedback persistence using ChromaDB.

Architecture:
  Per-user collection  : "fb_{sanitized_user_id}"
    → stores all reviews made by that specific user
    → used for Level 1 learning (recall_feedback) and session persistence

  Global collection    : FEEDBACK_GLOBAL_COLLECTION ("feedback_global")
    → mirrors every feedback record across all users
    → used by self_improve.py for statistical pattern analysis (Gate 1)

Thread safety:
  ChromaDB's upsert() is thread-safe at the collection level.
  No additional locking is required for write operations.
  feedback_id is deterministic (SHA256) so concurrent saves of the same
  record are idempotent — the later upsert simply overwrites the earlier one
  with identical data.

Multi-user isolation:
  Each user's reviews are stored in their own collection.
  The global collection aggregates across all users for pattern analysis
  but is never exposed to individual users through the UI.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from am_ais_assist.config import (
    FEEDBACK_GLOBAL_COLLECTION,
    FEEDBACK_MIN_NOT_OK,
    FEEDBACK_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAIR_SEPARATOR = " ||| "


def _sanitize_collection_name(user_id: str) -> str:
    """Convert a user_id into a valid ChromaDB collection name.

    ChromaDB collection names must:
    - Be 3–63 characters long
    - Start and end with an alphanumeric character
    - Contain only alphanumeric characters, underscores, or hyphens
    - Not contain two consecutive periods

    Strategy: replace any non-alphanumeric character with '_', prefix with
    'fb_', and truncate to 63 characters.
    """
    safe = re.sub(r"[^a-zA-Z0-9]", "_", user_id)
    name = f"fb_{safe}"
    return name[:63]


def _make_feedback_id(user_id: str, query_id: str, matched_id: str) -> str:
    """Deterministic feedback record ID.

    Using SHA256 of the three key fields ensures:
    - The same (user, query, match) triple always maps to the same record.
    - Concurrent saves of the same record are idempotent upserts.
    - IDs are fixed-length and ChromaDB-safe.
    """
    raw = f"{user_id}\x00{query_id}\x00{matched_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FeedbackStore
# ---------------------------------------------------------------------------


class FeedbackStore:
    """Manages feedback storage and retrieval for all users.

    Designed as a process-level singleton — instantiate once via
    ``@st.cache_resource`` in ``app.py`` and pass the instance around.

    All write methods are thread-safe because ChromaDB's ``upsert()`` is
    internally serialised per collection.
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency at module load time.
        # core._get_chroma_client() returns the process-level ChromaDB singleton
        # so we share the same client as the embedding index — no second client.
        from am_ais_assist.core import _get_chroma_client  # noqa: PLC0415

        self._get_client = _get_chroma_client
        logger.info("FeedbackStore initialised.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self, name: str):
        """Return (or create) a ChromaDB collection by name.

        Collections are created without an embedding_function because we
        always provide embeddings manually — the same pattern used by
        core.py for the base-file embedding index.
        """
        client = self._get_client()
        try:
            return client.get_collection(name)
        except Exception:  # noqa: BLE001
            return client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    def _collection_exists(self, name: str) -> bool:
        client = self._get_client()
        try:
            client.get_collection(name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _encode_pair(self, query_cleaned: str, matched_cleaned: str, model: Any) -> np.ndarray:
        """Encode a (query, match) pair into a single L2-normalised vector."""
        pair_text = query_cleaned + _PAIR_SEPARATOR + matched_cleaned
        raw = model.encode([pair_text], convert_to_numpy=True)
        normed: np.ndarray = normalize(raw, axis=1, norm="l2")
        return normed[0]

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def save_feedback(  # noqa: PLR0913
        self,
        *,
        user_id: str,
        user_name: str,
        session_id: str,
        query_id: str,
        matched_id: str,
        query_text: str,
        matched_text: str,
        query_cleaned: str,
        matched_cleaned: str,
        ai_score: float | str,
        ai_level: str,
        ai_remark: str,
        verdict: str,
        base_file_hash: str,
        check_file_hash: str,
        prompt_version: str,
        model: Any,
        user_remark: str = "",
    ) -> str:
        """Persist one reviewed result row to both user and global collections.

        Args:
            user_id:         Stable user identifier from auth headers.
            user_name:       Human-readable display name.
            session_id:      Current Streamlit session UUID.
            query_id:        Query_Object_Identifier value.
            matched_id:      Matched_Object_Identifier value.
            query_text:      Raw query sentence text.
            matched_text:    Raw matched sentence text.
            query_cleaned:   Preprocessed query text (used for embedding).
            matched_cleaned: Preprocessed matched text (used for embedding).
            ai_score:        Original AI similarity score (float or "N/A").
            ai_level:        Original AI similarity label.
            ai_remark:       Original AI remark.
            verdict:         User verdict: "OK" | "Not OK".
            base_file_hash:  SHA-256 of the base Excel file (for run identity).
            check_file_hash: SHA-256 of the check Excel file (for run identity).
            prompt_version:  Active prompt version string (e.g. "v1").
            model:           EmbeddingModel instance for pair encoding.
            user_remark:     Optional remark/comment from the user.

        Returns:
            feedback_id (str) — the deterministic SHA-256 record identifier.
        """
        # Normalize verdict to "Not Ok" for global self_improve consistency
        if verdict in ("Not OK", "Not Ok", "not ok"):
            verdict = "Not Ok"

        feedback_id = _make_feedback_id(user_id, query_id, matched_id)

        # Encode the pair — used for Level 1 semantic recall
        try:
            embedding = self._encode_pair(query_cleaned, matched_cleaned, model)
            embeddings_list = [embedding.tolist()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackStore: embedding failed (%s) — storing without embedding.", exc)
            embeddings_list = None

        metadata = {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "user_name": user_name,
            "session_id": session_id,
            "query_id": query_id,
            "matched_id": matched_id,
            "query_text": query_text[:500],      # truncate long texts for metadata storage
            "matched_text": matched_text[:500],
            "query_cleaned": query_cleaned[:300],
            "matched_cleaned": matched_cleaned[:300],
            "ai_score": str(ai_score),
            "ai_level": ai_level,
            "ai_remark": ai_remark[:300],
            "verdict": verdict,
            "user_remark": str(user_remark)[:200],
            "base_file_hash": base_file_hash[:32],
            "check_file_hash": check_file_hash[:32],
            "prompt_version": prompt_version,
            "timestamp": _now_iso(),
        }

        pair_text = query_cleaned + _PAIR_SEPARATOR + matched_cleaned

        upsert_kwargs: dict = {
            "ids": [feedback_id],
            "documents": [pair_text],
            "metadatas": [metadata],
        }
        if embeddings_list is not None:
            upsert_kwargs["embeddings"] = embeddings_list

        # Write to per-user collection
        try:
            user_col_name = _sanitize_collection_name(user_id)
            user_col = self._get_or_create_collection(user_col_name)
            user_col.upsert(**upsert_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("FeedbackStore: failed to save to user collection: %s", exc)

        # Write to global collection (pattern analysis source)
        try:
            global_col = self._get_or_create_collection(FEEDBACK_GLOBAL_COLLECTION)
            global_col.upsert(**upsert_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("FeedbackStore: failed to save to global collection: %s", exc)

        logger.debug(
            "FeedbackStore: saved feedback %s (user=%s, verdict=%s).",
            feedback_id[:8],
            user_id[:8],
            verdict,
        )
        return feedback_id

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def load_run_reviews(
        self,
        user_id: str,
        base_file_hash: str,
        check_file_hash: str,
    ) -> dict[str, dict[str, str]]:
        """Load prior reviews for a specific (user, base_file, check_file) combination.

        Called after the pipeline runs to pre-populate the review panel with
        any verdicts the user has already submitted for this exact file pair.

        Args:
            user_id:         Stable user identifier.
            base_file_hash:  First 32 chars of the base file SHA-256 hash.
            check_file_hash: First 32 chars of the check file SHA-256 hash.

        Returns:
            dict mapping query_id → {"verdict": verdict, "user_remark": remark} for all
            previously reviewed rows.  Returns ``{}`` when no prior reviews
            exist or the collection has not been created yet.
        """
        user_col_name = _sanitize_collection_name(user_id)
        if not self._collection_exists(user_col_name):
            return {}

        try:
            user_col = self._get_or_create_collection(user_col_name)
            results = user_col.get(
                where={
                    "$and": [
                        {"base_file_hash": {"$eq": base_file_hash[:32]}},
                        {"check_file_hash": {"$eq": check_file_hash[:32]}},
                    ]
                },
                include=["metadatas"],
            )
            reviews: dict[str, dict[str, str]] = {}
            for meta in results.get("metadatas", []):
                qid = meta.get("query_id", "")
                verdict = meta.get("verdict", "Pending")
                user_remark = meta.get("user_remark", "")
                if qid:
                    reviews[qid] = {"verdict": verdict, "user_remark": user_remark}
            logger.info(
                "FeedbackStore: loaded %d prior reviews for user %s on this file pair.",
                len(reviews),
                user_id[:8],
            )
            return reviews
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackStore: load_run_reviews failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Level 1 Learning — semantic recall
    # ------------------------------------------------------------------

    def recall_feedback(
        self,
        query_cleaned: str,
        matched_cleaned: str,
        user_id: str,
        model: Any,
    ) -> dict | None:
        """Check if this (query, match) pair has been reviewed before.

        Uses cosine similarity on pair embeddings.  Only returns a stored
        verdict when the similarity meets FEEDBACK_SIMILARITY_THRESHOLD
        (default 0.97) — tight enough to prevent false positives from
        superficially similar but semantically different pairs.

        Args:
            query_cleaned:   Preprocessed query text.
            matched_cleaned: Preprocessed matched text.
            user_id:         User identifier for scoped lookup.
            model:           EmbeddingModel for encoding the current pair.

        Returns:
            Metadata dict of the closest matching stored feedback, or
            ``None`` when no prior review meets the similarity threshold.
        """
        user_col_name = _sanitize_collection_name(user_id)
        if not self._collection_exists(user_col_name):
            return None

        try:
            embedding = self._encode_pair(query_cleaned, matched_cleaned, model)
            user_col = self._get_or_create_collection(user_col_name)

            # Check if this collection has any embeddings stored
            count = user_col.count()
            if count == 0:
                return None

            results = user_col.query(
                query_embeddings=[embedding.tolist()],
                n_results=1,
                include=["metadatas", "distances"],
            )

            if not results or not results.get("metadatas"):
                return None

            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            if not metadatas or not distances:
                return None

            # ChromaDB cosine distance = 1 - cosine_similarity
            # Convert distance back to similarity
            distance = float(distances[0])
            similarity = 1.0 - distance

            if similarity >= FEEDBACK_SIMILARITY_THRESHOLD:
                meta = metadatas[0]
                logger.debug(
                    "FeedbackStore: recall hit (similarity=%.4f) for user %s.",
                    similarity,
                    user_id[:8],
                )
                return meta

            return None

        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackStore: recall_feedback failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Pattern analysis (Gate 1 data source)
    # ------------------------------------------------------------------

    def get_feedback_statistics(self) -> dict:
        """Aggregate global feedback for statistical pattern analysis.

        Called exclusively by ``self_improve.run_gate1_statistical_analysis()``.
        Returns ONLY counts and percentages — never raw requirement text —
        to prevent prompt injection in downstream LLM calls.

        Returns:
            dict with keys:
              total_reviews (int)
              total_not_ok (int)
              min_threshold_met (bool)
              by_level (dict[str, int])       — Not Ok counts per AI level
              by_score_range (dict[str, int]) — Not Ok counts per score bucket
              by_prompt_version (dict[str, int]) — Not Ok counts per prompt version
        """
        if not self._collection_exists(FEEDBACK_GLOBAL_COLLECTION):
            return {
                "total_reviews": 0,
                "total_not_ok": 0,
                "min_threshold_met": False,
                "by_level": {},
                "by_score_range": {},
                "by_prompt_version": {},
            }

        try:
            global_col = self._get_or_create_collection(FEEDBACK_GLOBAL_COLLECTION)
            all_results = global_col.get(include=["metadatas"])
            all_meta = all_results.get("metadatas", [])

            total = len(all_meta)
            not_ok = [m for m in all_meta if str(m.get("verdict")).lower() in ("not ok", "not_ok")]

            # Group Not Ok verdicts by AI level
            by_level: dict[str, int] = {}
            for m in not_ok:
                lvl = m.get("ai_level", "Unknown")
                by_level[lvl] = by_level.get(lvl, 0) + 1

            # Group Not Ok verdicts by score range bucket
            score_buckets = {
                "0.00–0.50": 0,
                "0.50–0.65": 0,
                "0.65–0.80": 0,
                "0.80–0.95": 0,
                "0.95–1.00": 0,
            }
            for m in not_ok:
                try:
                    s = float(m.get("ai_score", 0))
                    if s < 0.50:
                        score_buckets["0.00–0.50"] += 1
                    elif s < 0.65:
                        score_buckets["0.50–0.65"] += 1
                    elif s < 0.80:
                        score_buckets["0.65–0.80"] += 1
                    elif s < 0.95:
                        score_buckets["0.80–0.95"] += 1
                    else:
                        score_buckets["0.95–1.00"] += 1
                except (ValueError, TypeError):
                    pass

            # Group by prompt version (helps detect regressions after deployment)
            by_prompt: dict[str, int] = {}
            for m in not_ok:
                pv = m.get("prompt_version", "unknown")
                by_prompt[pv] = by_prompt.get(pv, 0) + 1

            return {
                "total_reviews": total,
                "total_not_ok": len(not_ok),
                "min_threshold_met": len(not_ok) >= FEEDBACK_MIN_NOT_OK,
                "by_level": by_level,
                "by_score_range": {k: v for k, v in score_buckets.items() if v > 0},
                "by_prompt_version": by_prompt,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("FeedbackStore: get_feedback_statistics failed: %s", exc)
            return {
                "total_reviews": 0,
                "total_not_ok": 0,
                "min_threshold_met": False,
                "by_level": {},
                "by_score_range": {},
                "by_prompt_version": {},
            }

    def get_all_global_metadata(self) -> list[dict]:
        """Return all metadata records from the global feedback collection.

        Used by self_improve.py for holdout splitting and shadow testing.
        Callers must NOT pass the raw requirement text to any LLM —
        use aggregated statistics only.
        """
        if not self._collection_exists(FEEDBACK_GLOBAL_COLLECTION):
            return []
        try:
            global_col = self._get_or_create_collection(FEEDBACK_GLOBAL_COLLECTION)
            results = global_col.get(include=["metadatas"])
            return results.get("metadatas", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackStore: get_all_global_metadata failed: %s", exc)
            return []

    def get_by_ids(self, feedback_ids: list[str]) -> list[dict]:
        """Retrieve metadata for specific feedback record IDs.

        Used by self_improve.py to load the held-out set for shadow testing.
        """
        if not feedback_ids or not self._collection_exists(FEEDBACK_GLOBAL_COLLECTION):
            return []
        try:
            global_col = self._get_or_create_collection(FEEDBACK_GLOBAL_COLLECTION)
            results = global_col.get(ids=feedback_ids, include=["metadatas"])
            return results.get("metadatas", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackStore: get_by_ids failed: %s", exc)
            return []

    def get_by_prompt_version(self, prompt_version: str) -> list[dict]:
        """Return all feedback records tagged with a specific prompt version.

        Used by prompt_registry.py to monitor canary agreement rates.
        """
        if not self._collection_exists(FEEDBACK_GLOBAL_COLLECTION):
            return []
        try:
            global_col = self._get_or_create_collection(FEEDBACK_GLOBAL_COLLECTION)
            results = global_col.get(
                where={"prompt_version": {"$eq": prompt_version}},
                include=["metadatas"],
            )
            return results.get("metadatas", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackStore: get_by_prompt_version failed: %s", exc)
            return []

    def compute_agreement_rate(self, metadata_list: list[dict]) -> float:
        """Fraction of records where verdict == 'OK' (AI was right).

        Returns 0.0 when the list is empty.
        """
        if not metadata_list:
            return 0.0
        ok_count = sum(1 for m in metadata_list if str(m.get("verdict")).upper() == "OK")
        return ok_count / len(metadata_list)
