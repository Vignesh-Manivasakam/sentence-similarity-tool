"""
llm_service.py — LLM semantic analysis with proper system/user role separation,
robust JSON parsing, token-limit-aware batching, and thread-safe caching.

Key changes from original:
- System prompt sent as "system" role, sentence pairs as "user" role.
  GPT-4 class models follow system-role instructions more consistently.
- "Sentence Pairs:" header removed from system_prompt.txt and injected
  here in the user message, keeping the prompt file clean.
- JSON extraction uses re.sub instead of lstrip/rstrip (lstrip removes
  individual characters, not substrings — caused silent JSON parse errors).
- Token budget splits system tokens from pair tokens correctly so batching
  arithmetic is accurate.

Phase 5 additions:
- _call_llm_api() and get_llm_analysis_batch() accept an optional
  system_prompt parameter so the canary prompt version can be used for a
  specific session without touching the module-level SYSTEM_PROMPT constant.
  This ensures multi-user isolation: each session uses its own prompt version
  without affecting any other concurrent session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any

import filelock
import tiktoken
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from am_ais_assist.config import (
    ENABLE_PROMPT_CACHING,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_BATCH_TOKEN_LIMIT,
    LLM_CACHE_FILE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT_PATH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client — lazy-safe: initialised at module load but guarded so import never
# raises even when the key is missing.
# ---------------------------------------------------------------------------
client: OpenAI | None = None
try:
    if not LLM_API_KEY:
        logger.error("LLM API Key not configured. Set the NVIDIA_API_KEY environment variable.")
    else:
        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=30,
        )
        logger.info("LLM client initialised successfully for NVIDIA NIM.")
except Exception as exc:
    logger.error("Failed to initialise LLM client: %s", exc)

# ---------------------------------------------------------------------------
# System prompt — loaded once from file.
# "Sentence Pairs:" is intentionally NOT in the file; it is injected below
# as the opening line of every user message so the roles stay clean.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = ""
try:
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as fh:
        SYSTEM_PROMPT = fh.read().strip()
    logger.info("System prompt loaded from %s (%d chars).", SYSTEM_PROMPT_PATH, len(SYSTEM_PROMPT))
except FileNotFoundError:
    logger.error("System prompt file not found at: %s", SYSTEM_PROMPT_PATH)
    SYSTEM_PROMPT = "You are a requirements analyst. Return results as a JSON array."


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

# M-1: module-level tiktoken singleton — avoids recreating the encoder per call
_tiktoken_enc: Any | None = None
_tiktoken_lock = threading.Lock()


def _get_tiktoken_enc() -> Any:
    global _tiktoken_enc  # noqa: PLW0603
    if _tiktoken_enc is None:
        with _tiktoken_lock:
            if _tiktoken_enc is None:
                _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


def estimate_tokens(text: str) -> int:
    """Estimate token count via tiktoken (cl100k_base covers GPT-3.5/4)."""
    try:
        enc = _get_tiktoken_enc()
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return len(text) // 4


# Pre-compute system prompt token cost once so batching arithmetic is correct.
_SYSTEM_PROMPT_TOKENS: int = estimate_tokens(SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def compute_prompt_hash(system: str, user: str) -> str:
    """Hash both messages together so cache keys are fully unique."""
    combined = system + "\x00" + user  # null byte separator
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_user_llm_cache_file(user_session_id: str) -> str:
    from am_ais_assist.config import get_user_cache_dir

    return os.path.join(get_user_cache_dir(user_session_id), "llm_results_cache.json")


def load_llm_cache(user_session_id: str | None = None) -> dict:
    """Load cached LLM results for a specific user (or global fallback)."""
    cache_file = get_user_llm_cache_file(user_session_id) if user_session_id else LLM_CACHE_FILE
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load LLM cache '%s': %s", cache_file, exc)
        return {}


def save_llm_cache(cache: dict, user_session_id: str | None = None) -> None:
    """Save LLM results to user-specific cache file.

    .. deprecated::
        Prefer ``_append_to_user_cache()`` for concurrent-safe incremental writes.
        This function performs a full file replacement without a file lock and
        is retained only for backwards-compatibility with existing callers.
    """
    cache_file = get_user_llm_cache_file(user_session_id) if user_session_id else LLM_CACHE_FILE
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=4)
    except (TypeError, OSError) as exc:
        logger.error("Failed to save LLM cache: %s", exc)


def _append_to_user_cache(cache_key: str, response_data: dict, user_session_id: str) -> None:
    """
    Thread-safe append to the user's on-disk LLM cache.
    Uses filelock to prevent concurrent read-modify-write corruption (H-4).
    """
    if not user_session_id:
        return
    from am_ais_assist.config import get_user_cache_dir

    cache_file = os.path.join(get_user_cache_dir(user_session_id), "llm_results_cache.json")
    lock_path = cache_file + ".lock"
    try:
        with filelock.FileLock(lock_path, timeout=10):  # H-4: inter-process file lock
            current: dict = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, encoding="utf-8") as fh:
                        current = json.load(fh)
                except json.JSONDecodeError:
                    pass  # corrupted — start fresh
            current[cache_key] = response_data
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(current, fh, indent=4)
    except filelock.Timeout:
        logger.warning(
            "Could not acquire cache lock for user %s — skipping disk persist.", user_session_id
        )
    except Exception as exc:
        logger.error("Failed to persist LLM cache for user %s: %s", user_session_id, exc)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> str:
    """
    Strip markdown code fences from the LLM response before JSON parsing.

    lstrip/rstrip remove individual characters, not substrings — so the
    original code silently stripped 'j', 's', '`' etc. from valid JSON.
    This version uses a proper regex.
    """
    # Remove ```json ... ``` or ``` ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Core API call — system/user role split
# Phase 5: Added optional system_prompt parameter for canary prompt support.
# When system_prompt is provided it overrides the module-level SYSTEM_PROMPT
# for THIS call only — no other concurrent calls are affected.
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError)),  # C-2
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm_api(
    user_message: str,
    num_pairs: int,
    system_prompt: str | None = None,
) -> dict:
    """
    Make a single LLM API call with proper role separation.

    Args:
        user_message:  The sentence pairs formatted for the user turn.
        num_pairs:     Expected number of results for response validation.
        system_prompt: Optional override for the system prompt (used for
                       canary version routing). When None the module-level
                       SYSTEM_PROMPT constant is used.

    Returns:
        dict with keys 'results' (list) and 'tokens_used' (dict).
    """
    if client is None:  # C-6: null guard
        raise RuntimeError(
            "LLM client is not initialised. Set the GENAI_API_KEY environment variable."
        )

    # Use the provided system_prompt (canary) or fall back to the module default.
    effective_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
    effective_prompt_tokens = (
        estimate_tokens(effective_prompt) if system_prompt else _SYSTEM_PROMPT_TOKENS
    )

    content = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        # Token count: system + user combined (for usage reporting)
        prompt_tokens = effective_prompt_tokens + estimate_tokens(user_message)

        messages: list[dict] = [
            {"role": "system", "content": effective_prompt},
            {"role": "user", "content": user_message},
        ]

        # M-6: enable prompt caching when configured (reduces cost/latency)
        if ENABLE_PROMPT_CACHING:
            messages[0]["cache_control"] = {"type": "ephemeral"}  # type: ignore[index]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,  # M-6: configurable temperature (default 0.1)
            n=1,
        )

        content = response.choices[0].message.content or ""
        completion_tokens = estimate_tokens(content)

        # Safely extract JSON, handling markdown code fences
        json_str = _extract_json(content)
        analysis = json.loads(json_str)

        # Validate: must be a list with exactly num_pairs entries
        if not isinstance(analysis, list):
            raise ValueError(
                f"LLM returned a {type(analysis).__name__} instead of a JSON array. "
                "Check the system prompt Output Constraint section."
            )
        if len(analysis) != num_pairs:
            raise ValueError(
                f"Response length mismatch — expected {num_pairs}, got {len(analysis)}."
            )

        results = [
            {
                "Similarity_Score": item.get("score", "Error"),
                "Similarity_Level": item.get("relationship", "Parse Error"),
                "Remark": item.get("remark", "No specific difference analysis provided."),
            }
            for item in analysis
        ]
        return {
            "results": results,
            "tokens_used": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    except (json.JSONDecodeError, ValueError) as exc:
        # C-2: parsing/validation errors are not transient — return error sentinel
        error_type = type(exc).__name__
        logger.error(
            "LLM response parse error (%s): %s. Raw response: '%s'", error_type, exc, content
        )
        error_result = {
            "Similarity_Score": "Error",
            "Similarity_Level": f"{error_type} Error",
            "Remark": "Error during analysis.",
        }
        return {
            "results": [error_result] * num_pairs,
            "tokens_used": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    # APIConnectionError, RateLimitError, APIStatusError propagate to the @retry decorator


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def _build_user_message(batch_pairs: list[tuple[str, str]]) -> str:
    """
    Format sentence pairs as the user-turn message.
    The 'Sentence Pairs:' header is injected here (not in system_prompt.txt)
    so the system prompt stays focused on instructions only.
    """
    lines = ["Sentence Pairs:"]
    for i, (s1, s2) in enumerate(batch_pairs, start=1):
        lines.append(f"\nPair {i}:")
        lines.append(f'Sentence 1: "{s1}"')
        lines.append(f'Sentence 2: "{s2}"')
    return "\n".join(lines)


def _process_batch(
    batch_pairs: list[tuple[str, str]],
    user_session_id: str | None = None,
    cache_manager=None,
    system_prompt: str | None = None,
) -> dict:
    """Process one batch of sentence pairs through the LLM with cache lookup.

    Args:
        batch_pairs:     List of (query_cleaned, matched_cleaned) tuples.
        user_session_id: For per-user result caching.
        cache_manager:   GlobalCacheManager for cross-user caching.
        system_prompt:   Optional override (canary prompt).  When provided
                         the cache key incorporates the prompt text so canary
                         and stable results are stored separately.
    """
    user_message = _build_user_message(batch_pairs)
    # Include the effective system prompt in the cache key so canary results
    # are stored separately from the stable-version results.
    effective_system = system_prompt if system_prompt else SYSTEM_PROMPT
    cache_key = compute_prompt_hash(effective_system, user_message)

    # 1. Check global in-memory cache first
    if cache_manager:
        cached = cache_manager.get_llm_result(cache_key)
        if cached:
            logger.info(
                "Cache hit (global) for key %s... (%d pairs)",
                cache_key[:10],
                len(batch_pairs),
            )
            return cached

    # 2. API call on cache miss
    logger.info("Calling LLM for %d pairs (cache miss).", len(batch_pairs))
    response = _call_llm_api(user_message, len(batch_pairs), system_prompt=system_prompt)

    # 3. Cache only clean results (not error results)
    all_clean = all(
        not str(res.get("Similarity_Level", "")).endswith("Error")
        and res.get("Similarity_Level") not in ("LLM API Call Failed",)
        for res in response["results"]
    )
    if all_clean:
        if cache_manager:
            cache_manager.update_llm_result(cache_key, response)
        _append_to_user_cache(cache_key, response, user_session_id)

    return response


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_llm_analysis_batch(
    sentence_pairs: list[tuple[str, str]],
    user_session_id: str | None = None,
    cache_manager: Any | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Analyse sentence pairs with token-limit-aware batching and global caching.

    Splits the input into batches that fit within LLM_BATCH_TOKEN_LIMIT,
    accounting for the system prompt token cost separately from pair tokens.

    Phase 5 addition: ``system_prompt`` parameter for canary prompt support.
    When provided, this prompt is used instead of the module-level constant
    for this call only — all concurrent calls are unaffected (thread-safe).

    Args:
        sentence_pairs:  List of (query_text, matched_text) tuples.
        user_session_id: Used for per-user LLM result caching.
        cache_manager:   GlobalCacheManager for cross-user result reuse.
        system_prompt:   Optional canary system prompt override.

    Returns:
        dict with 'results' (one entry per input pair) and 'tokens_used'.
    """
    logger.info(
        "User identified: %s (session: %s)",
        user_session_id or "anonymous",
        user_session_id or "no-session",
    )

    if not client:
        err = {
            "Similarity_Score": "Error",
            "Similarity_Level": "LLM API Key Not Configured",
            "Remark": "API key missing.",
        }
        return {"results": [err] * len(sentence_pairs), "tokens_used": {}}

    all_results: list = [None] * len(sentence_pairs)
    total_tokens: dict = {"prompt_tokens": 0, "completion_tokens": 0}

    # Token budget: system prompt costs are fixed per call; only pair tokens
    # count toward the variable portion of the batch limit.
    effective_system = system_prompt if system_prompt else SYSTEM_PROMPT
    effective_system_tokens = (
        estimate_tokens(effective_system) if system_prompt else _SYSTEM_PROMPT_TOKENS
    )
    pair_token_budget = LLM_BATCH_TOKEN_LIMIT - effective_system_tokens
    if pair_token_budget <= 0:
        logger.error(
            "LLM_BATCH_TOKEN_LIMIT (%d) is smaller than the system prompt (%d tokens). "
            "Increase LLM_BATCH_TOKEN_LIMIT in config.",
            LLM_BATCH_TOKEN_LIMIT,
            effective_system_tokens,
        )
        pair_token_budget = 2000  # safe fallback

    current_batch_pairs: list = []
    current_batch_indices: list = []
    current_batch_tokens = 0  # tracks only pair tokens, not system prompt

    def _flush_batch() -> None:
        nonlocal current_batch_pairs, current_batch_indices, current_batch_tokens
        batch_response = _process_batch(
            current_batch_pairs, user_session_id, cache_manager, system_prompt=system_prompt
        )
        for j, res in enumerate(batch_response["results"]):
            all_results[current_batch_indices[j]] = res
        total_tokens["prompt_tokens"] += batch_response["tokens_used"].get("prompt_tokens", 0)
        total_tokens["completion_tokens"] += batch_response["tokens_used"].get(
            "completion_tokens", 0
        )
        current_batch_pairs = []
        current_batch_indices = []
        current_batch_tokens = 0

    for i, pair in enumerate(sentence_pairs):
        pair_text = f'\nPair {len(current_batch_pairs) + 1}:\nSentence 1: "{pair[0]}"\nSentence 2: "{pair[1]}"'
        pair_tokens = estimate_tokens(pair_text)

        if current_batch_pairs and current_batch_tokens + pair_tokens > pair_token_budget:
            _flush_batch()

        current_batch_pairs.append(pair)
        current_batch_indices.append(i)
        current_batch_tokens += pair_tokens

    if current_batch_pairs:
        _flush_batch()

    # Safety net: fill any None slots that somehow slipped through
    if None in all_results:
        logger.error("Missing results after batching — filling with error sentinels.")
        err = {
            "Similarity_Score": "Error",
            "Similarity_Level": "Processing Error",
            "Remark": "Result missing after batch processing.",
        }
        all_results = [r if r is not None else err for r in all_results]

    return {"results": all_results, "tokens_used": total_tokens}


# ---------------------------------------------------------------------------
# Agent Decision — binary YES/NO verification (for borderline section matches)
# ---------------------------------------------------------------------------

_AGENT_DECISION_PROMPT: str = ""
try:
    from am_ais_assist.config import AGENT_DECISION_PROMPT_PATH as _ADP

    with open(_ADP, encoding="utf-8") as _fh:
        _AGENT_DECISION_PROMPT = _fh.read().strip()
    logger.info("Agent decision prompt loaded (%d chars).", len(_AGENT_DECISION_PROMPT))
except FileNotFoundError:
    logger.warning("Agent decision prompt file not found — using inline fallback.")
    _AGENT_DECISION_PROMPT = (
        'Answer YES or NO. Output only: {"decision": "YES" | "NO", "reason": "<15 words max>"}'
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_agent_decision(question: str) -> dict:
    """Call Gemini (temp=0.0) to make a binary YES/NO decision.

    Args:
        question: The question to ask the agent (max ~200 words for efficiency).

    Returns:
        dict with keys ``decision`` ("YES" or "NO") and ``reason`` (str).
        Falls back to ``{"decision": "NO", "reason": "no client"}`` when the LLM
        client is not initialised, or ``{"decision": "NO", "reason": str(e)}`` on
        non-retryable errors.
    """
    if client is None:
        logger.warning("call_agent_decision: LLM client not initialised.")
        return {"decision": "NO", "reason": "no client"}

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _AGENT_DECISION_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            timeout=15,
            n=1,
        )
        raw = response.choices[0].message.content or ""
        json_str = _extract_json(raw)
        result = json.loads(json_str)
        decision = str(result.get("decision", "NO")).strip().upper()
        if decision not in ("YES", "NO"):
            decision = "NO"
        reason = str(result.get("reason", ""))
        logger.debug("Agent decision: %s — %s", decision, reason)
        return {"decision": decision, "reason": reason}
    except (APIConnectionError, RateLimitError, APIStatusError):
        raise  # let @retry handle transient errors
    except Exception as exc:  # noqa: BLE001
        logger.warning("call_agent_decision non-retryable error: %s", exc)
        return {"decision": "NO", "reason": str(exc)}


# ---------------------------------------------------------------------------
# Section Detector — classify spreadsheet column structure
# ---------------------------------------------------------------------------

_SECTION_DETECTION_PROMPT: str = ""
try:
    from am_ais_assist.config import SECTION_DETECTION_PROMPT_PATH as _SDP

    with open(_SDP, encoding="utf-8") as _fh:
        _SECTION_DETECTION_PROMPT = _fh.read().strip()
    logger.info("Section detection prompt loaded (%d chars).", len(_SECTION_DETECTION_PROMPT))
except FileNotFoundError:
    logger.warning("Section detection prompt file not found — using inline fallback.")
    _SECTION_DETECTION_PROMPT = (
        "Detect document section structure. Output strict JSON with key 'strategy' "
        "set to one of: object_type_column, hierarchy_depth, regex_on_text, no_structure."
    )

_SECTION_STRATEGY_FALLBACK: dict = {
    "strategy": "no_structure",
    "heading_column": None,
    "heading_values": [],
    "number_column": None,
    "depth_threshold": None,
    "text_column": None,
    "confidence": 0.0,
    "reason": "fallback — LLM call failed",
}


_KNOWN_STRATEGIES = frozenset(
    {"object_type_column", "hierarchy_depth", "regex_on_text", "no_structure"}
)


# FIX BUG 8: Removed reraise=False from this decorator.
# With reraise=False, tenacity returned None when all retries were exhausted
# (instead of re-raising the exception). The caller in preprocess.py then called
# raw.get("strategy", ...) on None → AttributeError crash.
# The except Exception block inside the function already catches all non-retryable
# errors and returns _SECTION_STRATEGY_FALLBACK safely. Transient API errors
# (APIConnectionError, RateLimitError, APIStatusError) will now propagate to
# detect_section_strategy() which falls back to no_structure strategy gracefully.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def call_section_detector(column_names: list, sample_rows: list) -> dict:
    """Call Gemini (temp=0.0) to detect the section structure strategy of a spreadsheet.

    Args:
        column_names: List of column name strings from the DataFrame.
        sample_rows:  List of dicts (up to MAX_SAMPLE_ROWS_FOR_DETECTION rows).

    Returns:
        Strategy dict with keys matching SectionStrategy dataclass fields.
        Falls back to ``_SECTION_STRATEGY_FALLBACK`` on any error.
    """
    if client is None:
        logger.warning("call_section_detector: LLM client not initialised.")
        return {"strategy": "no_structure", "confidence": 0.0, "reason": "no client"}

    try:
        cols_text = "Column names: " + ", ".join(str(c) for c in column_names)
        rows_lines = []
        for i, row in enumerate(sample_rows):
            # Truncate individual cell values to 80 chars to keep the prompt compact
            pairs = ", ".join(
                f"{k}={str(v)[:80]}" for k, v in row.items()
            )
            rows_lines.append(f"  Row {i + 1}: {pairs}")
        user_message = f"{cols_text}\nSample rows:\n" + "\n".join(rows_lines)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SECTION_DETECTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            n=1,
        )
        raw = response.choices[0].message.content or ""
        json_str = _extract_json(raw)
        result = json.loads(json_str)

        # Validate strategy is one of the 4 known values
        if result.get("strategy") not in _KNOWN_STRATEGIES:
            logger.warning(
                "call_section_detector: unknown strategy '%s' — defaulting to no_structure.",
                result.get("strategy"),
            )
            result["strategy"] = "no_structure"

        # Normalise and fill missing keys from fallback
        for key, default in _SECTION_STRATEGY_FALLBACK.items():
            if key not in result:
                result[key] = default
        # Ensure heading_values is always a list
        if not isinstance(result.get("heading_values"), list):
            result["heading_values"] = []

        logger.info(
            "Section strategy detected: %s confidence=%.2f",
            result.get("strategy"),
            float(result.get("confidence", 0)),
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("call_section_detector failed: %s", exc)
        import copy
        return copy.deepcopy(_SECTION_STRATEGY_FALLBACK)
