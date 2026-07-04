"""
pipeline.py — End-to-end similarity search pipeline orchestrator.

Phase 5 additions:
  - run_llm_analysis_phase() accepts optional feedback_store, user_id,
    active_prompt_text to support Level 1 feedback recall and canary
    prompt routing.
  - run_similarity_pipeline() accepts optional feedback_store and
    prompt_registry; computes base/check file hashes early and returns
    them as part of the result tuple for session persistence.
  - Return tuple extended to 11 values:
      (enriched_results, base_embeddings, user_embeddings,
       base_data, user_data, base_skipped, user_skipped, llm_tokens,
       base_file_hash, check_file_hash, active_prompt_version)

Multi-user safety:
  - All new parameters are optional with safe defaults — existing callers
    (tests, legacy code) continue to work without modification.
  - active_prompt_text is passed per-call through the stack so different
    concurrent sessions can use different prompt versions without any
    module-level state mutation.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize

from am_ais_assist.config import (
    DEFAULT_THRESHOLDS,
    LLM_ANALYSIS_MIN_THRESHOLD,
    LLM_PERFECT_MATCH_THRESHOLD,
    MAX_CONCURRENT_USERS,
    OUTPUT_DIR,
)
from am_ais_assist.core import (
    EMBEDDING_DIMENSION,
    _compute_file_hash,
    build_deleted_row,
    build_new_requirement_row,
    create_vector_index,
    match_sections,
    scoped_search,
    search_similar,
)
from am_ais_assist.core import load_model as core_load_model
from am_ais_assist.llm_service import get_llm_analysis_batch
from am_ais_assist.preprocess import build_hierarchy_tree, detect_section_strategy
from am_ais_assist.progress_manager import UnifiedProgressManager
from am_ais_assist.utils import excel_to_json, read_excel_dataframe

# ---------------------------------------------------------------------------
# Model — loaded once at startup and shared across all pipeline calls.
# @st.cache_resource ensures only one instance exists per Streamlit process.
# ---------------------------------------------------------------------------


@st.cache_resource
def get_model():
    """Load and cache the embedding model — called only once per process."""
    logging.info("get_model() called — loading embedding model.")
    return core_load_model()


# Lazy singletons — populated on the first call to run_similarity_pipeline.
# Keeping them as module-level attributes lets tests override them freely
# (e.g. `pl.model = MagicMock()`) without triggering any real API calls
# at import time.
model = None
tokenizer = None
_pipeline_init_lock = threading.Lock()  # M-4: prevent concurrent model initialisation
# M-8: limit concurrent pipeline executions to MAX_CONCURRENT_USERS
_active_pipelines_sem = threading.Semaphore(MAX_CONCURRENT_USERS)


def _ensure_model_loaded() -> None:
    """Initialise global model and tokenizer on first call (thread-safe, M-4)."""
    global model, tokenizer  # noqa: PLW0603
    if model is None:
        with _pipeline_init_lock:
            if model is None:
                model = get_model()
                tokenizer = model.tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_user_output_dir(user_session_id: str | None) -> str:
    """Return the output directory for this session, falling back to global."""
    if user_session_id:
        from am_ais_assist.config import get_user_output_dir as _cfg_output_dir

        return _cfg_output_dir(user_session_id)
    return OUTPUT_DIR


def _make_progress_callback(
    progress_manager: UnifiedProgressManager, prefix: str
) -> Callable[[int, int, str], None]:
    """Factory to create a progress callback for a pipeline phase."""

    def _callback(current: int, total: int, message: str) -> None:
        progress = current / total if total > 0 else 0
        progress_manager.update_phase_progress(progress, f"{prefix} {message}")

    return _callback


# ---------------------------------------------------------------------------
# Phase 4 — LLM analysis
# Phase 5 additions: feedback_store, user_id, active_prompt_text params
# ---------------------------------------------------------------------------


def run_llm_analysis_phase(  # noqa: PLR0912
    search_results: list,
    user_session_id: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cache_manager: Any | None = None,
    feedback_store: Any | None = None,
    user_id: str | None = None,
    active_prompt_text: str | None = None,
) -> tuple[list, dict]:
    """
    Enrich similarity search results with LLM semantic analysis.

    Phase 5 additions:
      feedback_store:     When provided, each row is checked against the
                          user's stored reviews before calling the LLM.
                          If a semantically identical pair was reviewed
                          before (Level 1 learning), the stored verdict is
                          used and the LLM call is skipped entirely.
      user_id:            Stable user identifier for feedback lookup.
      active_prompt_text: Optional canary prompt override. Passed through
                          to get_llm_analysis_batch() so multi-user canary
                          routing works without module-level state changes.

    Args:
        search_results:     List of dicts produced by search_similar().
        user_session_id:    Used for per-user LLM result caching.
        progress_callback:  callable(current, total, message).
        cache_manager:      GlobalCacheManager instance.
        feedback_store:     FeedbackStore instance (optional).
        user_id:            Stable user identifier (optional).
        active_prompt_text: Active (or canary) system prompt text (optional).

    Returns:
        Tuple of (enriched_results, token_usage_dict).
    """
    _empty_tokens: dict = {"prompt_tokens": 0, "completion_tokens": 0}

    if not search_results:
        logging.warning("run_llm_analysis_phase: no results to analyse.")
        return search_results, _empty_tokens

    try:
        effective_prompt = active_prompt_text
        if effective_prompt is None:
            from am_ais_assist.llm_service import SYSTEM_PROMPT as _default_prompt
            effective_prompt = _default_prompt
            
        if user_id:
            try:
                from am_ais_assist.skill_generator import get_user_skills_prompt_context
                user_rules = get_user_skills_prompt_context(user_id)
                if user_rules:
                    effective_prompt = effective_prompt + "\n\n" + user_rules
                    logging.info("Injected user-specific skills for user_id=%s into prompt", user_id)
            except Exception as _skills_exc:
                logging.warning("Failed to load user skills context: %s", _skills_exc)

        logging.info("Starting LLM analysis for %d results.", len(search_results))

        initial_df = pd.DataFrame(search_results)
        llm_results: list = []
        total_tokens: dict = {"prompt_tokens": 0, "completion_tokens": 0}
        total_rows = len(search_results)
        batch_size = 10
        total_batches = -(-total_rows // batch_size)  # ceiling division

        if progress_callback:
            progress_callback(0, total_batches, "Starting LLM analysis...")

        for batch_idx in range(0, total_rows, batch_size):
            batch_slice = search_results[batch_idx : batch_idx + batch_size]
            sentence_pairs = [
                (row["Query_Sentence_Cleaned_text"], row["Matched_Sentence_Cleaned_text"])
                for row in batch_slice
            ]
            batch_results: list = [None] * len(sentence_pairs)
            llm_pairs: list = []
            llm_indices: list = []

            for j, (_s1, _s2) in enumerate(sentence_pairs):
                score = batch_slice[j]["Similarity_Score"]

                # FIX BUG 4: Guard "New Requirement" and "Deleted" rows BEFORE
                # the score threshold checks — preserves labels set by core.py.
                current_level = batch_slice[j].get("Similarity_Level", "")
                if current_level in ("New Requirement", "Deleted"):
                    batch_results[j] = {
                        "Similarity_Score": score,
                        "Similarity_Level": current_level,
                        "Remark": batch_slice[j].get("Remark", ""),
                    }
                    continue

                # Phase 5 — Level 1 learning: check feedback store before LLM call.
                # If this exact (or semantically near-identical) pair was reviewed
                # before, use the stored verdict and skip the LLM entirely.
                if (
                    feedback_store is not None
                    and user_id is not None
                    and model is not None
                    and _s1
                    and _s2
                ):
                    try:
                        prior = feedback_store.recall_feedback(
                            query_cleaned=_s1,
                            matched_cleaned=_s2,
                            user_id=user_id,
                            model=model,
                        )
                        if prior is not None:
                            verdict = prior.get("verdict", "")
                            ai_level = prior.get("ai_level", current_level)
                            ai_remark = prior.get("ai_remark", "")
                            batch_results[j] = {
                                "Similarity_Score": score,
                                "Similarity_Level": ai_level,
                                "Remark": (
                                    f"[Recalled — previously reviewed as {verdict}] {ai_remark}"
                                ),
                            }
                            logging.debug(
                                "Feedback recall hit for pair at index %d (verdict=%s).",
                                batch_idx + j,
                                verdict,
                            )
                            continue
                    except Exception as _recall_exc:  # noqa: BLE001
                        logging.debug("Feedback recall failed (non-fatal): %s", _recall_exc)

                if score >= LLM_PERFECT_MATCH_THRESHOLD:
                    batch_results[j] = {
                        "Similarity_Score": 1.0,
                        "Similarity_Level": "Exact Match",
                        "Remark": "Exactly Matched",
                    }
                elif score < LLM_ANALYSIS_MIN_THRESHOLD:
                    batch_results[j] = {
                        "Similarity_Score": "N/A",
                        "Similarity_Level": "Below Threshold",
                        "Remark": "Analysis skipped — similarity score is below threshold.",
                    }
                else:
                    llm_pairs.append(sentence_pairs[j])
                    llm_indices.append(j)

            if llm_pairs:
                try:
                    batch_response = get_llm_analysis_batch(
                        llm_pairs,
                        user_session_id,
                        cache_manager,
                        system_prompt=effective_prompt,  # Injected prompt override (Q2)
                    )
                    for idx, result in zip(llm_indices, batch_response["results"], strict=False):
                        batch_results[idx] = result
                    total_tokens["prompt_tokens"] += batch_response["tokens_used"].get(
                        "prompt_tokens", 0
                    )
                    total_tokens["completion_tokens"] += batch_response["tokens_used"].get(
                        "completion_tokens", 0
                    )
                except Exception as exc:
                    logging.error(
                        "LLM analysis failed for batch %d: %s",
                        batch_idx // batch_size + 1,
                        exc,
                    )
                    error_result = {
                        "Similarity_Score": "Error",
                        "Similarity_Level": "Analysis Error",
                        "Remark": f"LLM analysis failed: {exc}",
                    }
                    for idx in llm_indices:
                        batch_results[idx] = error_result

            llm_results.extend(batch_results)

            current_batch = batch_idx // batch_size + 1
            if progress_callback:
                progress_callback(
                    current_batch,
                    total_batches,
                    f"LLM Analysis... {current_batch}/{total_batches}",
                )
            logging.info("LLM batch %d/%d complete.", current_batch, total_batches)

        # Merge LLM columns back onto the original search results
        llm_df = pd.DataFrame(llm_results, index=initial_df.index)
        enriched_df = pd.concat([initial_df, llm_df], axis=1)
        enriched_results = enriched_df.to_dict("records")

        logging.info("LLM analysis complete. Token usage: %s", total_tokens)
        return enriched_results, total_tokens

    except Exception as exc:
        logging.error("Error in LLM analysis phase: %s", exc, exc_info=True)
        _error_row = {
            "Similarity_Score": "Error",
            "Similarity_Level": "Analysis Error",
            "Remark": f"LLM phase failed: {exc}",
        }
        fallback = [{**row, **_error_row} for row in search_results]
        return fallback, _empty_tokens


# ---------------------------------------------------------------------------
# Main pipeline
# Phase 5: added feedback_store, prompt_registry optional params;
#          returns 11-tuple (original 8 + base_hash, check_hash, prompt_version)
# ---------------------------------------------------------------------------


def run_similarity_pipeline(  # noqa: PLR0913 PLR0915
    base_file,
    check_file,
    top_k: int,
    thresholds: dict | None,
    progress_manager: UnifiedProgressManager,
    base_id_col: str,
    base_text_col: str,
    check_id_col: str,
    check_text_col: str,
    base_meta_cols: list,
    check_meta_cols: list,
    user_session_id: str | None = None,
    cache_manager: Any | None = None,
    feedback_store: Any | None = None,
    prompt_registry: Any | None = None,
    user_id: str | None = None,
) -> tuple:
    """
    End-to-end similarity search pipeline.

    Phases:
        1. preprocessing     - parse and clean Excel files
        2. vector_index      - build / load ChromaDB collection
        3. similarity_search - exact-match + embedding search (hierarchical or flat)
        4. llm_analysis      - semantic enrichment via LLM (with feedback recall)

    Phase 5 additions:
        feedback_store:  FeedbackStore instance for Level 1 learning recall.
        prompt_registry: PromptRegistry instance for canary prompt routing.

    Returns:
        11-tuple: (enriched_results, base_embeddings, user_embeddings,
                   base_data, user_data, base_skipped, user_skipped, llm_tokens,
                   base_file_hash, check_file_hash, active_prompt_version)

        The last 3 values are NEW in Phase 5 and are used by app.py for:
          - Session persistence (pre-loading prior reviews)
          - Canary monitoring (recording which prompt version was used)
    """
    session_label = f"session {user_session_id}" if user_session_id else "global session"
    logging.info("Starting pipeline for %s.", session_label)

    # M-8: enforce concurrent user limit
    if not _active_pipelines_sem.acquire(blocking=False):
        raise RuntimeError(
            f"Server is at capacity ({MAX_CONCURRENT_USERS} concurrent analyses). "
            "Please wait a moment and try again."
        )
    _ensure_model_loaded()

    # Phase 5: resolve active prompt version for this session (canary routing)
    active_prompt_text: str | None = None
    active_prompt_version: str = "v1"
    if prompt_registry is not None:
        try:
            active_prompt_version = prompt_registry.get_active_version(user_session_id)
            active_prompt_text = prompt_registry.get_active_prompt_text(user_session_id)
            logging.info(
                "Prompt version for session %s: %s",
                (user_session_id or "anon")[:8],
                active_prompt_version,
            )
        except Exception as _pr_exc:  # noqa: BLE001
            logging.warning("Could not load prompt version (%s) — using default.", _pr_exc)

    # Compute file hashes early for session persistence and hierarchical path
    # _compute_file_hash handles its own seek(0) so no pre-seek needed.
    base_file_hash: str = ""
    check_file_hash: str = ""

    try:
        base_file_hash = _compute_file_hash(base_file)
        check_file_hash = _compute_file_hash(check_file)
    except Exception as _hash_exc:  # noqa: BLE001
        logging.warning("Could not compute file hashes (%s) — session persistence disabled.", _hash_exc)

    try:
        # ── Phase 1: Preprocessing ────────────────────────────────────────
        progress_manager.start_phase("preprocessing", "📄 Processing Excel files...")

        base_data, base_skipped = excel_to_json(
            base_file, tokenizer, base_id_col, base_text_col, base_meta_cols
        )
        progress_manager.update_phase_progress(0.5, "📄 Processing base file...")

        user_data, user_skipped = excel_to_json(
            check_file, tokenizer, check_id_col, check_text_col, check_meta_cols
        )
        progress_manager.update_phase_progress(1.0, "📄 File processing complete...")

        if not base_data or not user_data:
            raise ValueError("No valid data after preprocessing — check your files.")

        progress_manager.complete_phase("📄 Preprocessing complete!")

        # ── Phase 2: Vector index ─────────────────────────────────────────
        progress_manager.start_phase(
            "vector_index",
            f"🏗️ Building vector index ({len(base_data)} items)...",
        )

        _vector_index_progress = _make_progress_callback(progress_manager, "🏗️")

        vector_index, base_embeddings, base_data = create_vector_index(
            base_data,
            model,
            base_file,
            user_session_id,
            progress_callback=_vector_index_progress,
            cache_manager=cache_manager,
        )

        progress_manager.complete_phase("🏗️ Vector index ready!")

        # ── Phase 3: Similarity search ────────────────────────────────────
        progress_manager.start_phase("similarity_search", "🔍 Searching similarities...")

        internal_thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        _search_progress = _make_progress_callback(progress_manager, "🔍")

        base_df = read_excel_dataframe(base_file)
        new_df = read_excel_dataframe(check_file)

        search_results: list[dict] = []
        user_embeddings: np.ndarray = np.zeros(
            (len(user_data), EMBEDDING_DIMENSION), dtype=np.float32
        )
        _used_hierarchical = False

        if base_df is not None and new_df is not None:
            try:
                base_strategy = detect_section_strategy(base_df, file_hash=base_file_hash)
                new_strategy = detect_section_strategy(new_df, file_hash=check_file_hash)

                # FIX BUG 7: Normalise strategy column names after Excel rename
                if base_strategy.number_column == base_id_col:
                    base_strategy.number_column = "Object_Identifier"
                if new_strategy.number_column == check_id_col:
                    new_strategy.number_column = "Object_Identifier"

                base_tree = build_hierarchy_tree(base_data, base_strategy)
                new_tree = build_hierarchy_tree(user_data, new_strategy)

                if base_tree and new_tree:
                    _used_hierarchical = True
                    progress_manager.update_phase_progress(0.1, "🌲 Matching document sections...")

                    section_pairs = match_sections(base_tree, new_tree, model)

                    total_pairs = len(section_pairs)
                    for pair_idx, (base_sec, new_sec) in enumerate(section_pairs):
                        pair_frac = (pair_idx + 1) / max(total_pairs, 1)
                        progress_manager.update_phase_progress(
                            0.1 + pair_frac * 0.85,
                            f"🔍 Comparing section {pair_idx + 1}/{total_pairs}...",
                        )

                        base_items = base_tree.get(base_sec, []) if base_sec else []
                        new_items = new_tree.get(new_sec, []) if new_sec else []

                        if base_sec and new_sec:
                            section_results = scoped_search(
                                new_items, base_items, model, internal_thresholds
                            )
                        elif new_sec and not base_sec:
                            section_results = [build_new_requirement_row(n) for n in new_items]
                        else:
                            section_results = [build_deleted_row(b) for b in base_items]

                        search_results.extend(section_results)

                    # Handle ungrouped items (not captured by any section)
                    grouped_new_ids = {
                        entry["Object_Identifier"]
                        for items in new_tree.values()
                        for entry in items
                    }
                    ungrouped_new = [
                        u for u in user_data
                        if u["Object_Identifier"] not in grouped_new_ids
                    ]
                    if ungrouped_new:
                        _fallback_results, _ = search_similar(
                            ungrouped_new,
                            vector_index,
                            base_data,
                            top_k,
                            internal_thresholds,
                            model,
                            progress_callback=None,
                        )
                        search_results.extend(_fallback_results)

                    # FIX BUG 5: compute user embeddings for the 3D visualisation
                    if user_data:
                        try:
                            user_texts = [e["Cleaned_Text"] for e in user_data]
                            raw_user_embs = model.encode(user_texts, convert_to_numpy=True)
                            user_embeddings = normalize(raw_user_embs, axis=1, norm="l2")
                            logging.info(
                                "User embeddings computed: %d vectors.", len(user_data)
                            )
                        except Exception as _emb_exc:
                            logging.warning(
                                "Could not compute user embeddings: %s", _emb_exc
                            )

                    logging.info(
                        "Hierarchical search complete: %d results from %d section pairs.",
                        len(search_results),
                        total_pairs,
                    )

            except Exception as _hier_exc:
                logging.warning(
                    "Hierarchical search failed (%s) — falling back to flat search.", _hier_exc
                )
                _used_hierarchical = False
                search_results = []

        # ── Flat search fallback ──────────────────────────────────────────
        if not _used_hierarchical:
            search_results, user_embeddings = search_similar(
                user_data,
                vector_index,
                base_data,
                top_k,
                internal_thresholds,
                model,
                progress_callback=_search_progress,
            )

        progress_manager.complete_phase(f"🔍 Found {len(search_results)} matches!")

        # ── Phase 4: LLM analysis ─────────────────────────────────────────
        progress_manager.start_phase("llm_analysis", "🧠 Running LLM analysis...")

        _llm_progress = _make_progress_callback(progress_manager, "🧠")

        # Phase 5: extract user_id for feedback recall
        # user_session_id is a UUID; user_id from app.py auth headers is the
        # stable user identifier. We pass user_id if provided, otherwise fallback.
        _user_id_for_recall = user_id if user_id else user_session_id

        try:
            enriched_results, llm_tokens = run_llm_analysis_phase(
                search_results,
                user_session_id,
                progress_callback=_llm_progress,
                cache_manager=cache_manager,
                feedback_store=feedback_store,
                user_id=_user_id_for_recall,
                active_prompt_text=active_prompt_text,
            )
            total_used = llm_tokens["prompt_tokens"] + llm_tokens["completion_tokens"]
            progress_manager.complete_phase(f"🧠 LLM analysis complete! ({total_used} tokens used)")
        except Exception as exc:
            logging.error("LLM analysis phase raised unexpectedly: %s", exc)
            progress_manager.update_phase_progress(
                1.0, "⚠️ LLM analysis failed — showing similarity scores only."
            )
            enriched_results = search_results
            llm_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

        # ── Done ──────────────────────────────────────────────────────────
        progress_manager.complete_all()
        logging.info("Pipeline completed successfully for %s.", session_label)

        return (
            enriched_results,
            base_embeddings,
            user_embeddings,
            base_data,
            user_data,
            base_skipped,
            user_skipped,
            llm_tokens,
            base_file_hash,       # Phase 5 — for session persistence
            check_file_hash,      # Phase 5 — for session persistence
            active_prompt_version,  # Phase 5 — for canary monitoring
        )

    except Exception as exc:
        error_msg = f"Pipeline error for {session_label}: {exc}"
        logging.error(error_msg, exc_info=True)
        progress_manager.progress_bar.progress(0.0, text=f"❌ {error_msg[:80]}")
        raise
    finally:
        _active_pipelines_sem.release()  # M-8: always release the semaphore
