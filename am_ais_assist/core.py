"""
core.py — Embedding generation, ChromaDB vector storage, and similarity search.

Replaces FAISS flat-file approach with ChromaDB for:
- Persistent, multi-user isolated collections
- Built-in metadata storage (no separate .npy files)
- Native cross-user collection reuse via shared collection names
- Eliminates manual cache file management and path-replacement bugs

Core search logic (exact-match shortcut, hierarchy tie-breaker,
search_similar) is UNCHANGED from the FAISS version.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chromadb
import faiss
import numpy as np
import tiktoken
from chromadb.config import Settings
from openai import OpenAI
from sklearn.preprocessing import normalize

from am_ais_assist.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_SERVER_HOST,
    CHROMA_SERVER_PORT,
    EMBEDDING_API_KEY,
    EMBEDDING_API_VERSION,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    ITEM_NEW_REQ_THRESHOLD,
    PLACEHOLDER_DELETED,
    PLACEHOLDER_DELETED_QUERY,
    PLACEHOLDER_NO_MATCH,
    SECTION_AUTO_MATCH_THRESHOLD,
    SECTION_UNCERTAIN_THRESHOLD,
)
from am_ais_assist.postprocess import highlight_word_differences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global progress callback (thread-local to fix the multi-user race condition
# that existed in the FAISS version with a plain global variable)
# ---------------------------------------------------------------------------
import threading

_tl = threading.local()


def set_embedding_progress_callback(cb: Any) -> None:
    """Register a per-thread progress callback (current, total) → None."""
    _tl.embedding_progress_cb = cb


def _emit_progress(current: int, total: int) -> None:
    cb = getattr(_tl, "embedding_progress_cb", None)
    if cb is not None:
        try:
            cb(current, total)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Progress callback raised: %s", exc)


# ---------------------------------------------------------------------------
# OpenAI embedding client (lazy-initialised to avoid import-time side-effects)
# ---------------------------------------------------------------------------
_embedding_client: OpenAI | None = None
_embedding_client_lock = threading.Lock()  # C-3: thread-safe singleton


def _get_embedding_client() -> OpenAI:
    global _embedding_client  # noqa: PLW0603
    if _embedding_client is None:
        with _embedding_client_lock:  # C-3: double-checked locking
            if _embedding_client is None:
                if not EMBEDDING_API_KEY:
                    raise RuntimeError(
                        "Embedding API key not configured. "
                        "Set the NVIDIA_API_KEY environment variable."
                    )
                _embedding_client = OpenAI(
                    api_key=EMBEDDING_API_KEY,
                    base_url=EMBEDDING_BASE_URL,
                )
                logger.info("OpenAI embedding client initialised for NVIDIA NIM.")
    return _embedding_client


# ---------------------------------------------------------------------------
# ChromaDB client (module-level singleton — one client, many collections)
# ---------------------------------------------------------------------------
_chroma_client: chromadb.ClientAPI | None = None
_chroma_client_lock = threading.Lock()  # C-3 + H-2: thread-safe singleton and rmtree guard


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return the singleton ChromaDB persistent or HTTP client.

    Self-heals when the on-disk database was created by an older ChromaDB
    version (e.g. pre-1.0) whose tenant schema is incompatible with the
    installed version.  In that case the stale directory is wiped and the
    client is reinitialised from scratch — embeddings will be recomputed on
    the next pipeline run.
    """
    global _chroma_client  # noqa: PLW0603
    if _chroma_client is None:
        with _chroma_client_lock:  # C-3: double-checked locking; H-2: protects shutil.rmtree
            if _chroma_client is None:
                if CHROMA_SERVER_HOST:
                    try:
                        _chroma_client = chromadb.HttpClient(
                            host=CHROMA_SERVER_HOST,
                            port=CHROMA_SERVER_PORT,
                            settings=Settings(anonymized_telemetry=False),
                        )
                        logger.info(
                            "ChromaDB HTTP client initialised connecting to %s:%s",
                            CHROMA_SERVER_HOST,
                            CHROMA_SERVER_PORT,
                        )
                    except Exception as exc:
                        logger.error("Failed to initialize ChromaDB HTTP client: %s", exc)
                        raise
                else:
                    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                    try:
                        _chroma_client = chromadb.PersistentClient(
                            path=CHROMA_PERSIST_DIR,
                            settings=Settings(anonymized_telemetry=False),
                        )
                        logger.info("ChromaDB persistent client initialised at %s", CHROMA_PERSIST_DIR)
                    except Exception as exc:  # noqa: BLE001
                        # ChromaDB ≥1.0 changed the tenant schema; the error message
                        # contains "tenant" when the existing SQLite database is stale.
                        if "tenant" in str(exc).lower():
                            logger.warning(
                                "ChromaDB tenant schema mismatch — wiping stale directory at %s and "
                                "reinitialising.  Embeddings will be recomputed on the next run. "
                                "Original error: %s",
                                CHROMA_PERSIST_DIR,
                                exc,
                            )
                            import shutil

                            shutil.rmtree(
                                CHROMA_PERSIST_DIR, ignore_errors=True
                            )  # H-2: protected by lock
                            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                            _chroma_client = chromadb.PersistentClient(
                                path=CHROMA_PERSIST_DIR,
                                settings=Settings(anonymized_telemetry=False),
                            )
                            logger.info(
                                "ChromaDB client reinitialised (clean) at %s", CHROMA_PERSIST_DIR
                            )
                        else:
                            raise
    return _chroma_client


# ---------------------------------------------------------------------------
# MockTokenizer — identical to FAISS version, no changes needed
# ---------------------------------------------------------------------------


class MockTokenizer:
    """
    Compatibility shim so the preprocessing pipeline can call
    tokenizer.encode() without a HuggingFace model present.
    """

    def __init__(self) -> None:
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            logger.warning("tiktoken unavailable — using character approximation.")
            self.encoding = None
        self.sep_token_id = 102

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(text, str):
            logger.warning("Non-string input to tokenizer: %s", type(text))
            return []
        if self.encoding:
            try:
                tokens = self.encoding.encode(text, disallowed_special=())
                if add_special_tokens:
                    tokens = [101, *tokens, 102]
                return tokens
            except Exception as exc:  # noqa: BLE001
                logger.warning("tiktoken failed (%s) — falling back.", exc)
        n = len(text) // 4 + (2 if add_special_tokens else 0)
        return list(range(n))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        # C-1: Cannot reconstruct text from integer IDs without the actual vocabulary.
        # Raising here signals the caller to use character-level truncation instead,
        # preventing the literal "[TRUNCATED_TO_N_TOKENS]" string from being embedded.
        raise NotImplementedError(
            "MockTokenizer cannot decode token IDs back to text. "
            "Use character-based truncation for OpenAI embeddings."
        )


# ---------------------------------------------------------------------------
# EmbeddingModel — parallel batching, deduplication (unchanged logic)
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSION = 1536  # text-embedding-3-small


class EmbeddingModel:
    """OpenAI embedding wrapper with parallel batching and input deduplication."""

    def __init__(self, batch_size: int = 256, max_workers: int = 4) -> None:
        self.tokenizer = MockTokenizer()
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _embed_batch(self, batch: list[str]) -> list[np.ndarray]:
        client = _get_embedding_client()
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
            extra_query={"api-version": EMBEDDING_API_VERSION},
        )
        return [np.array(r.embedding) for r in response.data]

    def encode(
        self,
        texts: list[str] | str,
        *,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,  # noqa: ARG002
        batch_size: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([]) if convert_to_numpy else []

        effective_batch = batch_size or self.batch_size
        unique_texts = list(dict.fromkeys(texts))
        batches = [
            unique_texts[i : i + effective_batch]
            for i in range(0, len(unique_texts), effective_batch)
        ]

        logger.info(
            "Encoding %d unique texts in %d batches (batch_size=%d)",
            len(unique_texts),
            len(batches),
            effective_batch,
        )

        text_to_embedding: dict[str, np.ndarray] = {}
        completed = 0
        _emit_progress(0, len(batches))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._embed_batch, b): b for b in batches}
            for fut in as_completed(futures):
                batch = futures[fut]
                try:
                    embeddings = fut.result()
                    for t, emb in zip(batch, embeddings, strict=False):
                        text_to_embedding[t] = emb
                except Exception as exc:  # noqa: BLE001
                    logger.error("Batch embedding failed: %s", exc)
                    # C-5: unit-vector sentinel instead of zeros — zeros produce NaN after L2 norm
                    sentinel = np.full(
                        EMBEDDING_DIMENSION, 1.0 / np.sqrt(EMBEDDING_DIMENSION), dtype=np.float32
                    )
                    for t in batch:
                        text_to_embedding[t] = sentinel.copy()
                    logger.warning(
                        "%d texts in failed batch use sentinel embedding — results may be inaccurate.",
                        len(batch),
                    )
                completed += 1
                _emit_progress(completed, len(batches))

        all_embeddings = [text_to_embedding[t] for t in texts]
        return np.array(all_embeddings) if convert_to_numpy else all_embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return EMBEDDING_DIMENSION


# ---------------------------------------------------------------------------
# Model loader (cached by pipeline.py via @st.cache_resource)
# ---------------------------------------------------------------------------


def load_model() -> EmbeddingModel:
    try:
        _get_embedding_client()  # Fail-fast if key missing
        model = EmbeddingModel()
        logger.info("EmbeddingModel ready.")
        return model
    except Exception as exc:
        logger.error("FATAL: could not load embedding model: %s", exc)
        raise RuntimeError(f"Could not load embedding model: {exc}") from exc


# ---------------------------------------------------------------------------
# File hashing (unchanged)
# ---------------------------------------------------------------------------


def _compute_file_hash(file_obj: Any) -> str:
    sha256 = hashlib.sha256()
    file_obj.seek(0)
    for chunk in iter(lambda: file_obj.read(4096), b""):
        sha256.update(chunk)
    file_obj.seek(0)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# ChromaDB collection helpers
# Collection naming convention:
#   "ais_<first16_of_hash>"  — shared across users for the same file content
#   This replaces GlobalCacheManager.embedding_map entirely.
# ---------------------------------------------------------------------------

_COLLECTION_PREFIX = "ais_"


def _collection_name(file_hash: str) -> str:
    """Deterministic, cross-user collection name from file content hash."""
    return f"{_COLLECTION_PREFIX}{file_hash[:32]}"


def _collection_exists(file_hash: str) -> bool:
    client = _get_chroma_client()
    name = _collection_name(file_hash)
    try:  # L-4: O(1) direct lookup instead of O(n) list_collections()
        client.get_collection(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_from_collection(
    file_hash: str,
) -> tuple[Any, np.ndarray] | tuple[None, None]:
    """
    Load embeddings from ChromaDB and rebuild an IndexFlatIP in memory.

    ChromaDB is used purely as a persistent embedding store.
    The FAISS IndexFlatIP is always rebuilt from those embeddings so
    search remains exact brute-force inner product — identical to the
    original implementation.

    Also validates embedding dimension so a model-version change never
    silently produces wrong scores (a bug in the original FAISS version).

    Returns (faiss_index, embeddings_array) or (None, None) on failure.
    """
    client = _get_chroma_client()
    name = _collection_name(file_hash)
    try:
        collection = client.get_collection(name)
        result = collection.get(include=["embeddings"])
        embeddings = np.array(result["embeddings"], dtype=np.float32)

        if embeddings.shape[0] == 0:
            logger.warning("Collection %s is empty — regenerating.", name)
            return None, None

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            logger.warning(
                "Dimension mismatch in collection %s: stored=%d expected=%d — regenerating.",
                name,
                embeddings.shape[1],
                EMBEDDING_DIMENSION,
            )
            client.delete_collection(name)
            return None, None

        # Rebuild IndexFlatIP in memory — exact brute-force inner product,
        # same as the original. Embeddings are already L2-normalised so
        # inner product == cosine similarity.
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        index.add(embeddings)

        logger.info(
            "Loaded %d embeddings from ChromaDB '%s' → IndexFlatIP rebuilt.",
            len(embeddings),
            name,
        )
        return index, embeddings

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load collection %s: %s", name, exc)
        return None, None


def _build_and_store_collection(
    file_hash: str,
    data: list[dict],
    model: EmbeddingModel,
    progress_callback: Any = None,
) -> tuple[Any, np.ndarray]:
    """
    Generate embeddings via OpenAI, persist in ChromaDB, and return an
    IndexFlatIP built from those embeddings.

    ChromaDB stores: raw embeddings + metadata (for cross-user reuse).
    IndexFlatIP: built in memory for exact brute-force search.

    Returns (faiss_index, normalised_embeddings).
    """
    texts = [entry["Cleaned_Text"] for entry in data]
    if not texts:
        raise ValueError("No texts available for embedding generation.")

    # Wire progress callback into thread-local slot
    if progress_callback:

        def _cb(current: int, total: int) -> None:
            progress_callback(
                0.2 + (current / total) * 0.7 if total else 0.2,
                1.0,
                f"🔗 Generating embeddings… {current}/{total} batches",
            )

        set_embedding_progress_callback(_cb)
    else:
        set_embedding_progress_callback(None)

    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    finally:
        set_embedding_progress_callback(None)

    if progress_callback:
        progress_callback(0.85, 1.0, "🏗️ Building IndexFlatIP…")

    # L2-normalise so inner product == cosine similarity (same as original)
    normalised: np.ndarray = normalize(embeddings, axis=1, norm="l2")

    # ── Build IndexFlatIP — exact brute-force, unchanged from original ────
    dim = normalised.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalised)
    logger.info("IndexFlatIP built: %d vectors, dim=%d.", len(normalised), dim)

    if progress_callback:
        progress_callback(0.90, 1.0, "💾 Persisting to ChromaDB…")

    # ── Persist embeddings + metadata to ChromaDB for future reuse ────────
    chroma_client = _get_chroma_client()
    name = _collection_name(file_hash)

    try:
        chroma_client.delete_collection(name)
        logger.info("Deleted stale collection %s before rebuild.", name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Collection %s not found for deletion (will create fresh): %s", name, exc)

    collection = chroma_client.create_collection(
        name=name,
        # Store as "ip" space so if ChromaDB is ever queried directly the
        # distance metric is consistent with IndexFlatIP semantics.
        metadata={
            "hnsw:space": "ip",
            "embedding_model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
        },
    )

    # P-4: use upsert() instead of add() for idempotency.
    # ChromaDB's default batch limit is 41666; we stay well under with batches of 500.
    batch_size = 500
    total_batches = -(-len(data) // batch_size)
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        batch_emb = normalised[i : i + batch_size]
        collection.upsert(
            ids=[str(entry["Object_Identifier"]) for entry in batch_data],
            embeddings=batch_emb.tolist(),
            documents=[entry["Cleaned_Text"] for entry in batch_data],
            metadatas=[
                {
                    "Object_Identifier": str(entry["Object_Identifier"]),
                    "Original_Text": entry.get("Original_Text", ""),
                    "Hierarchy": entry.get("Hierarchy") or "",
                    "Truncated": str(entry.get("Truncated", False)),
                }
                for entry in batch_data
            ],
        )
        logger.info("Stored batch %d/%d in ChromaDB.", i // batch_size + 1, total_batches)

    if progress_callback:
        progress_callback(1.0, 1.0, "✅ Index ready!")

    logger.info("Collection %s created: %d embeddings, dim=%d.", name, len(data), dim)
    return index, normalised


# ---------------------------------------------------------------------------
# Per-file-hash build locks — prevent concurrent builds of same collection (C-4)
# ---------------------------------------------------------------------------
_build_locks: dict[str, threading.Lock] = {}
_build_locks_meta = threading.Lock()


def _get_build_lock(file_hash: str) -> threading.Lock:
    """Return a per-hash lock so concurrent uploads of the same file don't race."""
    with _build_locks_meta:
        if file_hash not in _build_locks:
            _build_locks[file_hash] = threading.Lock()
        return _build_locks[file_hash]


# ---------------------------------------------------------------------------
# Public API: create_faiss_index → renamed to create_vector_index
# Signature is IDENTICAL to old create_faiss_index so pipeline.py needs
# only a one-line import change.
# ---------------------------------------------------------------------------


def create_vector_index(  # noqa: PLR0913
    data: list[dict],
    model: EmbeddingModel,
    base_file_obj: Any,
    user_session_id: str | None = None,  # kept for signature compatibility  # noqa: ARG001
    progress_callback: Any = None,
    cache_manager: Any = None,  # kept for signature compatibility, no longer needed  # noqa: ARG001
) -> tuple[Any, np.ndarray, list[dict]]:
    """
    Create or load a ChromaDB collection for the base file.

    Drop-in replacement for create_faiss_index().
    Returns (collection, embeddings_array, data) — same tuple shape.
    """
    if progress_callback:
        progress_callback(0, 3, "🔍 Computing file hash…")

    file_hash = _compute_file_hash(base_file_obj)

    with _get_build_lock(file_hash):  # C-4: serialize concurrent builds for the same file
        if _collection_exists(file_hash):
            if progress_callback:
                progress_callback(1, 3, "📦 Found existing collection — loading…")
            index, embeddings = _load_from_collection(file_hash)
            if index is not None and embeddings is not None:
                if progress_callback:
                    progress_callback(3, 3, "📦 Loaded from ChromaDB cache → IndexFlatIP ready")
                return index, embeddings, data
            logger.warning("Collection load failed — regenerating.")

        if progress_callback:
            progress_callback(2, 3, "⚡ Generating new embeddings…")

        collection, embeddings = _build_and_store_collection(
            file_hash, data, model, progress_callback
        )

    if progress_callback:
        progress_callback(3, 3, "🎉 Index generation complete")

    return collection, embeddings, data


# Backward-compat alias so pipeline.py import works with zero change
create_faiss_index = create_vector_index


# ---------------------------------------------------------------------------
# User cache files helper (kept for app.py debug panel compatibility)
# ---------------------------------------------------------------------------


def get_user_cache_files(user_session_id: str) -> dict[str, str]:  # noqa: ARG001
    """Returns stub paths — ChromaDB manages its own storage."""
    return {
        "chromadb_dir": CHROMA_PERSIST_DIR,
        "note": "ChromaDB manages persistence internally.",
    }


# ---------------------------------------------------------------------------
# Hierarchy tie-breaker (UNCHANGED from FAISS version)
# ---------------------------------------------------------------------------


def _parse_hierarchy(hstr: str | None) -> list[int]:
    if not hstr:
        return []
    parts = hstr.replace("-", ".").split(".")
    return [int(p) for p in parts if p.isdigit()]


def _hierarchy_distance(qh: str | None, bh: str | None) -> tuple[int, float]:
    q = _parse_hierarchy(qh)
    b = _parse_hierarchy(bh)
    if not q or not b:
        return (0, float("inf"))
    # Count longest common prefix — stop at first mismatch
    lcp = 0
    for i, (x, y) in enumerate(zip(q, b, strict=False)):
        if x != y:
            break
        lcp = i + 1
    diff = abs(q[lcp] - b[lcp]) if lcp < min(len(q), len(b)) else abs(len(q) - len(b))
    return (-lcp, diff)


def choose_by_hierarchy(query_entry: dict, candidates: list[dict]) -> dict:
    qh = query_entry.get("Hierarchy")
    if not qh or not candidates:
        return candidates[0]
    best, best_key = None, (float("inf"), float("inf"))
    for c in candidates:
        key = _hierarchy_distance(qh, c.get("Hierarchy"))
        if key < best_key:
            best, best_key = c, key
    return best if best else candidates[0]


# ---------------------------------------------------------------------------
# search_similar — uses IndexFlatIP.search() for exact brute-force inner product
# Uses index.search() — exact brute-force IndexFlatIP
# ---------------------------------------------------------------------------


def search_similar(  # noqa: PLR0913, PLR0912, PLR0915
    user_data: list[dict],
    index: Any,  # faiss.IndexFlatIP — exact brute-force inner product
    base_data: list[dict],
    top_k: int,
    thresholds: dict,
    model: EmbeddingModel,
    progress_callback: Any = None,
) -> tuple[list[dict], np.ndarray]:
    logger.info(
        "Starting similarity search: %d queries vs %d base items",
        len(user_data),
        len(base_data),
    )
    results: list[dict] = []
    total_phases = 3

    if progress_callback:
        progress_callback(0, total_phases, "Initialising similarity search…")

    text_to_candidates: dict[str, list[dict]] = {}
    for b in base_data:
        text_to_candidates.setdefault(b["Cleaned_Text"], []).append(b)

    exact_matches: dict[int, list[dict]] = {}
    remaining_user_data: list[dict] = []
    remaining_indices: list[int] = []

    # ── Phase 1: exact string match ───────────────────────────────────────
    if progress_callback:
        progress_callback(1, total_phases, f"Exact match pass ({len(user_data)} queries)…")

    for i, user_entry in enumerate(user_data):
        candidates = text_to_candidates.get(user_entry["Cleaned_Text"], [])
        if candidates:
            chosen = (
                candidates[0]
                if len(candidates) == 1
                else choose_by_hierarchy(user_entry, candidates)
            )
            q_hl, m_hl = highlight_word_differences(
                user_entry["Original_Text"], chosen["Original_Text"]
            )
            item = _build_result_item(user_entry, chosen, 1.0, "Exact Match", q_hl, m_hl)
            exact_matches[i] = [item]
        else:
            remaining_user_data.append(user_entry)
            remaining_indices.append(i)

    exact_count = len(exact_matches)
    logger.info(
        "Phase 1: %d exact matches, %d need embedding search",
        exact_count,
        len(remaining_user_data),
    )

    # ── Phase 2: IndexFlatIP — exact brute-force inner product ────────────
    if progress_callback:
        progress_callback(
            2, total_phases, f"Embedding search ({len(remaining_user_data)} queries)…"
        )

    remaining_embeddings_array: np.ndarray | None = None

    if remaining_user_data:
        remaining_texts = [e["Cleaned_Text"] for e in remaining_user_data]
        raw_embeddings = model.encode(remaining_texts, convert_to_numpy=True)
        remaining_embeddings_array = normalize(raw_embeddings, axis=1, norm="l2")

        # IndexFlatIP.search — exact O(n), scores are cosine similarities
        # because both query and base vectors are L2-normalised.
        D, I = index.search(remaining_embeddings_array, top_k)

        for idx, rem_idx in enumerate(remaining_indices):
            user_entry = remaining_user_data[idx]
            exact_matches[rem_idx] = []

            candidate_items = [
                (float(D[idx][rank]), base_data[I[idx][rank]])
                for rank in range(top_k)
                if I[idx][rank] < len(base_data)
            ]

            grouped: dict[str, list[tuple[float, dict]]] = {}
            for sc, m in candidate_items:
                grouped.setdefault(m["Cleaned_Text"], []).append((sc, m))

            for cleaned_text, returned_list in grouped.items():
                rep_score = max(sc for sc, _ in returned_list)
                candidates_for_text = text_to_candidates.get(
                    cleaned_text, [m for _, m in returned_list]
                )
                chosen = (
                    choose_by_hierarchy(user_entry, candidates_for_text)
                    if len(candidates_for_text) > 1
                    else candidates_for_text[0]
                )
                label = _classify(rep_score, thresholds)
                q_hl, m_hl = highlight_word_differences(
                    user_entry["Original_Text"], chosen["Original_Text"]
                )
                item = _build_result_item(
                    user_entry, chosen, round(rep_score, 4), label, q_hl, m_hl
                )
                exact_matches[rem_idx].append(item)

        logger.info(
            "Phase 2 complete: %d queries via IndexFlatIP (exact brute-force).",
            len(remaining_user_data),
        )
    else:
        logger.info("Phase 2 skipped — all queries are exact string matches.")
        if progress_callback:
            progress_callback(2, total_phases, "Embedding search skipped (all exact matches)")

    # ── Phase 3: collect results ──────────────────────────────────────────
    if progress_callback:
        progress_callback(3, total_phases, "Collecting results…")

    for i in range(len(user_data)):
        if i in exact_matches:
            results.extend(exact_matches[i])

    full_user_embeddings = np.zeros((len(user_data), model.get_sentence_embedding_dimension()))
    if remaining_user_data and remaining_embeddings_array is not None:
        for idx, rem_idx in enumerate(remaining_indices):
            full_user_embeddings[rem_idx] = remaining_embeddings_array[idx]

    logger.info(
        "Search complete: %d exact, %d via IndexFlatIP, %d total results.",
        exact_count,
        len(remaining_user_data),
        len(results),
    )
    if progress_callback:
        progress_callback(total_phases, total_phases, f"Done — {len(results)} results")

    return results, full_user_embeddings


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_EXACT_MATCH_EPSILON: float = 1e-6


def _classify(score: float, thresholds: dict) -> str:
    if abs(score - 1.0) < _EXACT_MATCH_EPSILON:
        return "Exact Match"
    if score >= thresholds["most"]:
        return "Most Similar"
    if score >= thresholds["moderate"]:
        return "Moderately Similar"
    return "No Match"


def _build_result_item(  # noqa: PLR0913
    user_entry: dict,
    chosen: dict,
    score: float,
    label: str,
    query_highlighted: str,
    match_highlighted: str,
) -> dict:
    item = {
        "Query_Object_Identifier": user_entry["Object_Identifier"],
        "Query_Sentence": user_entry["Original_Text"],
        "Query_Sentence_Cleaned_text": user_entry["Cleaned_Text"],
        "Query_Sentence_Highlighted": query_highlighted,
        "Matched_Object_Identifier": chosen["Object_Identifier"],
        "Matched_Sentence": chosen["Original_Text"],
        "Matched_Sentence_Cleaned_text": chosen["Cleaned_Text"],
        "Matched_Sentence_Highlighted": match_highlighted,
        "Similarity_Score": score,
        "Similarity_Level": label,
    }
    _exclude = {"Object_Identifier", "Original_Text", "Cleaned_Text", "Truncated", "Hierarchy"}
    for key, value in user_entry.items():
        if key not in _exclude:
            item[f"Query_{key}"] = value
    for key, value in chosen.items():
        if key not in _exclude:
            item[f"Matched_{key}"] = value
    return item


# ---------------------------------------------------------------------------
# Hierarchical Section-Aware Comparison — Skills 3, 4, 5
# ---------------------------------------------------------------------------


def match_sections(
    base_tree: dict[str, list[dict]],
    new_tree: dict[str, list[dict]],
    model: EmbeddingModel,
) -> list[tuple[str | None, str | None]]:
    """Skill 3 — Map base sections to new sections using cosine similarity.

    High-confidence pairs (≥ SECTION_AUTO_MATCH_THRESHOLD) are accepted
    automatically.  Borderline pairs (≥ SECTION_UNCERTAIN_THRESHOLD) are
    verified with a Gemini YES/NO call.  Unmatched sections are included as
    (base_section, None) or (None, new_section) pairs.

    Args:
        base_tree: section_title → [entries] for the base specification.
        new_tree:  section_title → [entries] for the new specification.
        model:     EmbeddingModel for computing section-title embeddings.

    Returns:
        List of (base_section_title | None, new_section_title | None) pairs.
    """
    from am_ais_assist.llm_service import call_agent_decision

    base_keys = list(base_tree.keys())
    new_keys = list(new_tree.keys())

    if not base_keys or not new_keys:
        return [(b, None) for b in base_keys] + [(None, n) for n in new_keys]

    # Embed section titles
    all_titles = base_keys + new_keys
    raw_embs = model.encode(all_titles, convert_to_numpy=True)
    norm_embs = normalize(raw_embs, axis=1, norm="l2")

    base_embs = norm_embs[: len(base_keys)]
    new_embs = norm_embs[len(base_keys) :]

    # Cosine similarity matrix (base × new)
    sim_matrix = base_embs @ new_embs.T  # shape (|base|, |new|)

    matched_base: set[int] = set()
    matched_new: set[int] = set()
    pairs: list[tuple[str | None, str | None]] = []

    # Greedy best-match pass
    # Sort all (base_i, new_j) pairs by descending score
    candidates = sorted(
        [
            (float(sim_matrix[bi, nj]), bi, nj)
            for bi in range(len(base_keys))
            for nj in range(len(new_keys))
        ],
        reverse=True,
    )

    for score, bi, nj in candidates:
        if bi in matched_base or nj in matched_new:
            continue

        if score >= SECTION_AUTO_MATCH_THRESHOLD:
            pairs.append((base_keys[bi], new_keys[nj]))
            matched_base.add(bi)
            matched_new.add(nj)
        elif score >= SECTION_UNCERTAIN_THRESHOLD:
            question = (
                f'Are these two section headings from different specification versions '
                f'describing the same topic?\n'
                f'Heading A: "{base_keys[bi]}"\n'
                f'Heading B: "{new_keys[nj]}"\n'
                f'Answer YES if they cover the same subject area, NO otherwise.'
            )
            verdict = call_agent_decision(question)
            if verdict.get("decision") == "YES":
                pairs.append((base_keys[bi], new_keys[nj]))
                matched_base.add(bi)
                matched_new.add(nj)
            else:
                logger.debug(
                    "Section mismatch (%.2f): '%s' vs '%s' — %s",
                    score,
                    base_keys[bi][:40],
                    new_keys[nj][:40],
                    verdict.get("reason", ""),
                )
        else:
            # FIX BUG 1: was `break` — which exited the entire loop and skipped all
            # remaining candidate pairs, even high-scoring ones for other sections.
            # Changed to `continue` so only this low-scoring pair is skipped.
            continue

    # Unmatched base sections → deleted
    for bi, bk in enumerate(base_keys):
        if bi not in matched_base:
            pairs.append((bk, None))

    # Unmatched new sections → entirely new
    for nj, nk in enumerate(new_keys):
        if nj not in matched_new:
            pairs.append((None, nk))

    logger.info(
        "match_sections: %d pairs (%d auto, %d unmatched-base, %d unmatched-new)",
        len(pairs),
        len(matched_base),
        sum(1 for b, n in pairs if b and not n),
        sum(1 for b, n in pairs if not b and n),
    )
    return pairs


def build_new_requirement_row(new_item: dict) -> dict:
    """Build a result row for a requirement that exists only in the new specification."""
    q_hl, _ = highlight_word_differences(new_item.get("Original_Text", ""), "")
    item = {
        "Query_Object_Identifier": new_item.get("Object_Identifier", ""),
        "Query_Sentence": new_item.get("Original_Text", ""),
        "Query_Sentence_Cleaned_text": new_item.get("Cleaned_Text", ""),
        "Query_Sentence_Highlighted": q_hl,
        "Matched_Object_Identifier": "",
        "Matched_Sentence": PLACEHOLDER_NO_MATCH,
        "Matched_Sentence_Cleaned_text": "",
        "Matched_Sentence_Highlighted": PLACEHOLDER_NO_MATCH,
        "Similarity_Score": 0.0,
        "Similarity_Level": "New Requirement",
        "Remark": "This requirement does not exist in the base specification.",
    }
    _exclude = {"Object_Identifier", "Original_Text", "Cleaned_Text", "Truncated", "Hierarchy"}
    for key, value in new_item.items():
        if key not in _exclude:
            item[f"Query_{key}"] = value
    return item


def build_deleted_row(base_item: dict) -> dict:
    """Build a result row for a requirement that exists only in the base specification."""
    _, m_hl = highlight_word_differences("", base_item.get("Original_Text", ""))
    item = {
        "Query_Object_Identifier": "",
        "Query_Sentence": PLACEHOLDER_DELETED_QUERY,
        "Query_Sentence_Cleaned_text": "",
        "Query_Sentence_Highlighted": PLACEHOLDER_DELETED_QUERY,
        "Matched_Object_Identifier": base_item.get("Object_Identifier", ""),
        "Matched_Sentence": base_item.get("Original_Text", ""),
        "Matched_Sentence_Cleaned_text": base_item.get("Cleaned_Text", ""),
        "Matched_Sentence_Highlighted": m_hl,
        "Similarity_Score": 0.0,
        "Similarity_Level": "Deleted",
        "Remark": "This requirement has been removed in the new specification.",
    }
    _exclude = {"Object_Identifier", "Original_Text", "Cleaned_Text", "Truncated", "Hierarchy"}
    for key, value in base_item.items():
        if key not in _exclude:
            item[f"Matched_{key}"] = value
    return item


def scoped_search(
    new_section: list[dict],
    base_section: list[dict],
    model: EmbeddingModel,
    thresholds: dict,
) -> list[dict]:
    """Skill 4 — Scoped requirement comparison within a matched section pair.

    Phase A: Exact string match (score = 1.0).
    Phase B: FAISS embedding search; borderline results [threshold, threshold+0.15]
             are verified with a Gemini YES/NO call.
    Phase C: Any base items not claimed by any new item → ``build_deleted_row()``.

    FIX BUG 2: Previously called model.encode([q_text]) once per item inside the
    loop, making N sequential API calls for N items. Now all non-exact-match items
    are batch-encoded before the loop using a single model.encode() call, then
    looked up from a pre-built dict during the loop. This is O(1) per loop
    iteration instead of O(N) API calls.

    Args:
        new_section:  List of new-spec requirement entries for this section.
        base_section: List of base-spec requirement entries for this section.
        model:        EmbeddingModel.
        thresholds:   Standard thresholds dict (keys: exact, most, moderate).

    Returns:
        List of result dicts ready for postprocessing.
    """
    from am_ais_assist.llm_service import call_agent_decision

    if not new_section:
        return [build_deleted_row(b) for b in base_section]
    if not base_section:
        return [build_new_requirement_row(n) for n in new_section]

    results: list[dict] = []
    claimed_base_ids: set[str] = set()

    # Build local FAISS index over base_section
    base_texts = [e["Cleaned_Text"] for e in base_section]
    raw_base = model.encode(base_texts, convert_to_numpy=True)
    norm_base: np.ndarray = normalize(raw_base, axis=1, norm="l2")

    dim = norm_base.shape[1]
    local_index = faiss.IndexFlatIP(dim)
    local_index.add(norm_base)

    text_to_base: dict[str, list[dict]] = {}
    for b in base_section:
        text_to_base.setdefault(b["Cleaned_Text"], []).append(b)

    # FIX BUG 2: Pre-compute embeddings for all non-exact-match items in one
    # batch call before entering the loop. This replaces N individual
    # model.encode([single_text]) calls with one model.encode(list_of_texts).
    items_needing_embed = [
        item for item in new_section if item["Cleaned_Text"] not in text_to_base
    ]
    embed_map: dict[str, np.ndarray] = {}
    if items_needing_embed:
        embed_texts = [e["Cleaned_Text"] for e in items_needing_embed]
        raw_batch = model.encode(embed_texts, convert_to_numpy=True)
        norm_batch: np.ndarray = normalize(raw_batch, axis=1, norm="l2")
        for i, item in enumerate(items_needing_embed):
            embed_map[item["Cleaned_Text"]] = norm_batch[i]

    borderline_upper = thresholds.get("moderate", 0.7) + 0.15

    for new_item in new_section:
        q_text = new_item["Cleaned_Text"]

        # Phase A — exact string match
        if q_text in text_to_base:
            candidates = text_to_base[q_text]
            chosen = choose_by_hierarchy(new_item, candidates)
            q_hl, m_hl = highlight_word_differences(new_item["Original_Text"], chosen["Original_Text"])
            row = _build_result_item(new_item, chosen, 1.0, "Exact Match", q_hl, m_hl)
            results.append(row)
            claimed_base_ids.add(str(chosen["Object_Identifier"]))
            continue

        # Phase B — embedding search using pre-computed embedding from embed_map
        # (FIX BUG 2: no longer calls model.encode here — uses pre-built dict)
        q_norm = embed_map.get(q_text)
        if q_norm is None:
            # Safety fallback: should not happen, but encode on-demand if missing
            logger.warning("embed_map miss for '%s' — encoding individually.", q_text[:40])
            raw_single = model.encode([q_text], convert_to_numpy=True)
            q_norm = normalize(raw_single, axis=1, norm="l2")[0]

        q_norm_2d = q_norm.reshape(1, -1)
        D, I = local_index.search(q_norm_2d, 1)
        score = float(D[0][0])
        best_base = base_section[int(I[0][0])]

        if score < ITEM_NEW_REQ_THRESHOLD:
            results.append(build_new_requirement_row(new_item))
            continue

        moderate_threshold = thresholds.get("moderate", 0.7)

        # Borderline — ask Gemini
        if moderate_threshold <= score <= borderline_upper:
            question = (
                f'Are these two requirements semantically equivalent '
                f'(same intent, same constraint)?\n'
                f'Requirement A: "{new_item.get("Original_Text", "")[:300]}"\n'
                f'Requirement B: "{best_base.get("Original_Text", "")[:300]}"\n'
                f'Answer YES only if they impose the same constraint.'
            )
            verdict = call_agent_decision(question)
            if verdict.get("decision") != "YES":
                results.append(build_new_requirement_row(new_item))
                continue

        label = _classify(score, thresholds)
        q_hl, m_hl = highlight_word_differences(
            new_item["Original_Text"], best_base["Original_Text"]
        )
        row = _build_result_item(new_item, best_base, round(score, 4), label, q_hl, m_hl)
        results.append(row)
        claimed_base_ids.add(str(best_base["Object_Identifier"]))

    # Phase C — deleted requirements (unclaimed base items)
    for base_item in base_section:
        if str(base_item["Object_Identifier"]) not in claimed_base_ids:
            results.append(build_deleted_row(base_item))

    return results
