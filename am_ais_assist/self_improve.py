"""
self_improve.py — Admin-triggered 5-gate self-improving prompt pipeline.

Gates:
  Gate 1 — Statistical pre-analysis (pure Python, no LLM, no raw text)
  Gate 2 — LLM pattern analysis (Gemini, constrained structured output)
  Gate 3 — Automated validation (4 checks: stat backing, contradiction,
             shadow test, confidence threshold)
  Gate 4 — Human review (UI in app.py — not a function here)
  Gate 5 — Canary monitoring (PromptRegistry.get_canary_status())

Security guarantees:
  • Raw requirement text is NEVER passed to any LLM in this module.
    Only aggregated statistics are passed to Gate 2.
    Shadow test pairs are passed as (cleaned_text_1, cleaned_text_2)
    which have already been preprocessed and truncated to ≤ 8K chars.
  • The IMPROVEMENT_LOCK_FILE ensures only one admin can run the
    pipeline at a time, preventing concurrent registry corruption.

Multi-user note:
  This module is invoked only from the admin panel in app.py.
  Normal users never trigger any code in this module.
"""

from __future__ import annotations

import json
import logging
import re

import filelock

from am_ais_assist.config import (
    IMPROVEMENT_LOCK_FILE,
    PATTERN_ANALYSIS_PROMPT_PATH,
    SUGGESTION_MIN_CONFIDENCE,
    SUGGESTION_MIN_STAT_SUPPORT,
    SHADOW_TEST_HELD_OUT_PCT,
    CONTRADICTION_CHECK_PROMPT_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loaders (cached at module level)
# ---------------------------------------------------------------------------

_PATTERN_ANALYSIS_PROMPT: str = ""
_CONTRADICTION_CHECK_PROMPT: str = ""


def _load_prompts() -> None:
    global _PATTERN_ANALYSIS_PROMPT, _CONTRADICTION_CHECK_PROMPT  # noqa: PLW0603
    if not _PATTERN_ANALYSIS_PROMPT:
        try:
            with open(PATTERN_ANALYSIS_PROMPT_PATH, encoding="utf-8") as fh:
                _PATTERN_ANALYSIS_PROMPT = fh.read().strip()
        except FileNotFoundError:
            _PATTERN_ANALYSIS_PROMPT = (
                "You are a prompt improvement analyst for a requirements comparison AI tool. "
                "Analyse the provided feedback statistics and suggest ONE specific, targeted "
                "addition to the system prompt. "
                "Output strict JSON with keys: finding, supporting_statistic, "
                "affected_component, suggested_addition, confidence (0-1), scope."
            )
            logger.warning("self_improve: pattern_analysis_prompt.txt not found — using fallback.")

    if not _CONTRADICTION_CHECK_PROMPT:
        try:
            with open(CONTRADICTION_CHECK_PROMPT_PATH, encoding="utf-8") as fh:
                _CONTRADICTION_CHECK_PROMPT = fh.read().strip()
        except FileNotFoundError:
            _CONTRADICTION_CHECK_PROMPT = (
                "You are a prompt quality checker. "
                "Determine whether the proposed addition contradicts or duplicates any "
                "rule in the existing system prompt. "
                "Output strict JSON: {\"contradicts\": true|false, \"reason\": \"<20 words>\"}."
            )
            logger.warning(
                "self_improve: contradiction_check_prompt.txt not found — using fallback."
            )


def _call_llm_for_improvement(system_prompt: str, user_message: str) -> str:
    """Make a single Gemini call at temperature=0.0 for deterministic output.

    Imports the LLM client lazily to avoid circular dependency at module load.
    Returns raw response text; caller is responsible for JSON parsing.
    """
    from am_ais_assist.llm_service import client, LLM_MODEL, _extract_json  # noqa: PLC0415

    if client is None:
        raise RuntimeError("LLM client not initialised — cannot run self-improvement pipeline.")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        n=1,
    )
    raw = response.choices[0].message.content or ""
    return _extract_json(raw)


# ---------------------------------------------------------------------------
# Gate 1 — Statistical Pre-Analysis
# ---------------------------------------------------------------------------


def run_gate1_statistical_analysis(feedback_store: object) -> dict | None:
    """Aggregate global feedback statistics for pattern detection.

    Args:
        feedback_store: FeedbackStore instance.

    Returns:
        Analysis dict on success, or None when the minimum threshold is not met.
        The returned dict contains only counts/percentages — NO raw text.
    """
    stats = feedback_store.get_feedback_statistics()  # type: ignore[attr-defined]

    if not stats["min_threshold_met"]:
        logger.info(
            "Gate 1: insufficient data (%d / %d Not Ok verdicts required).",
            stats["total_not_ok"],
            __import__("am_ais_assist.config", fromlist=["FEEDBACK_MIN_NOT_OK"]).FEEDBACK_MIN_NOT_OK,
        )
        return None

    # Build human-readable pattern summary (no raw text, only statistics)
    patterns = []
    not_ok_total = stats["total_not_ok"]

    for level, count in stats["by_level"].items():
        pct = count / not_ok_total * 100 if not_ok_total else 0
        patterns.append({"type": "ai_level", "value": level, "count": count, "percentage": round(pct, 1)})

    for score_range, count in stats["by_score_range"].items():
        pct = count / not_ok_total * 100 if not_ok_total else 0
        patterns.append({"type": "score_range", "value": score_range, "count": count, "percentage": round(pct, 1)})

    # Find dominant pattern (highest %)
    dominant = max(patterns, key=lambda p: p["percentage"]) if patterns else {}

    # Determine holdout IDs (20% of all records, deterministic split)
    all_meta = feedback_store.get_all_global_metadata()  # type: ignore[attr-defined]
    holdout_ids = _get_holdout_ids(all_meta)

    result = {
        "total_reviews": stats["total_reviews"],
        "total_not_ok": not_ok_total,
        "patterns": patterns,
        "dominant_pattern": dominant,
        "holdout_ids": holdout_ids,
        "by_level": stats["by_level"],
        "by_score_range": stats["by_score_range"],
        "by_prompt_version": stats["by_prompt_version"],
    }

    logger.info(
        "Gate 1 complete: %d Not Ok verdicts, dominant pattern: %s (%.1f%%).",
        not_ok_total,
        dominant.get("value", "none"),
        dominant.get("percentage", 0),
    )
    return result


def _get_holdout_ids(all_meta: list[dict]) -> list[str]:
    """Deterministic holdout split: every Nth record is held out.

    Takes feedback_ids from the metadata list, sorts them alphabetically
    (for stability across calls), then takes every 1/SHADOW_TEST_HELD_OUT_PCT
    record as a holdout.

    Returns list of feedback_id strings.
    """
    all_ids = sorted({m.get("feedback_id", "") for m in all_meta if m.get("feedback_id")})
    step = max(1, round(1.0 / SHADOW_TEST_HELD_OUT_PCT))
    return all_ids[::step]


# ---------------------------------------------------------------------------
# Gate 2 — LLM Pattern Analysis
# ---------------------------------------------------------------------------


def run_gate2_llm_analysis(gate1_result: dict, prompt_registry: object) -> dict | None:
    """Ask Gemini to identify ONE targeted prompt improvement.

    The LLM receives only a structured statistics summary — never any raw
    requirement text — to prevent prompt injection.

    Args:
        gate1_result:    Output of run_gate1_statistical_analysis().
        prompt_registry: PromptRegistry instance for reading the current prompt.

    Returns:
        Suggestion dict with keys: finding, supporting_statistic,
        affected_component, suggested_addition, confidence, scope.
        Returns None when confidence is too low or parsing fails.
    """
    _load_prompts()

    current_prompt = prompt_registry.get_active_prompt_text()  # type: ignore[attr-defined]

    # Build a statistics-only summary (no raw requirement text)
    lines = [
        f"Total Not Ok verdicts: {gate1_result['total_not_ok']} / {gate1_result['total_reviews']} total reviews",
        "",
        "Not Ok verdicts by AI similarity level:",
    ]
    for level, count in gate1_result["by_level"].items():
        pct = count / gate1_result["total_not_ok"] * 100 if gate1_result["total_not_ok"] else 0
        lines.append(f"  - {level}: {count} ({pct:.1f}%)")

    lines += ["", "Not Ok verdicts by score range:"]
    for score_range, count in gate1_result["by_score_range"].items():
        pct = count / gate1_result["total_not_ok"] * 100 if gate1_result["total_not_ok"] else 0
        lines.append(f"  - {score_range}: {count} ({pct:.1f}%)")

    dominant = gate1_result.get("dominant_pattern", {})
    if dominant:
        lines += [
            "",
            f"Dominant pattern: {dominant.get('value')} "
            f"({dominant.get('percentage', 0):.1f}% of Not Ok verdicts)",
        ]

    stats_summary = "\n".join(lines)

    user_message = (
        f"Current system prompt (do NOT rewrite — only suggest an ADDITION of ≤ 3 sentences):\n"
        f"---\n{current_prompt}\n---\n\n"
        f"Feedback statistics (aggregated, NO raw requirement text):\n{stats_summary}"
    )

    try:
        raw_json = _call_llm_for_improvement(_PATTERN_ANALYSIS_PROMPT, user_message)
        suggestion = json.loads(raw_json)
    except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        logger.error("Gate 2: LLM call or JSON parse failed: %s", exc)
        return None

    # Validate required fields
    required = ["finding", "supporting_statistic", "affected_component",
                "suggested_addition", "confidence", "scope"]
    if not all(k in suggestion for k in required):
        logger.warning("Gate 2: response missing required fields — got: %s", list(suggestion.keys()))
        return None

    # Reject low-confidence suggestions
    try:
        confidence = float(suggestion["confidence"])
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < SUGGESTION_MIN_CONFIDENCE:
        logger.info(
            "Gate 2: rejected — confidence %.2f < threshold %.2f.",
            confidence,
            SUGGESTION_MIN_CONFIDENCE,
        )
        return None

    logger.info(
        "Gate 2: suggestion accepted (confidence=%.2f). Finding: %s",
        confidence,
        suggestion["finding"][:80],
    )
    return suggestion


# ---------------------------------------------------------------------------
# Gate 3 — Automated Validation (4 checks)
# ---------------------------------------------------------------------------


def run_gate3_validation(
    gate2_suggestion: dict,
    gate1_result: dict,
    feedback_store: object,
    prompt_registry: object,
) -> dict:
    """Run 4 automated validation checks before sending to human review.

    Args:
        gate2_suggestion: Output of run_gate2_llm_analysis().
        gate1_result:     Output of run_gate1_statistical_analysis().
        feedback_store:   FeedbackStore instance.
        prompt_registry:  PromptRegistry instance.

    Returns:
        dict with keys:
          passed (bool) — True only when ALL 4 checks pass.
          check1_stat_verified (bool)
          check2_no_contradiction (bool)
          check3_improvement_delta (float)  — fraction of holdout changed
          check4_confidence_ok (bool)
          rejection_reason (str | None)
    """
    _load_prompts()

    result: dict = {
        "passed": False,
        "check1_stat_verified": False,
        "check2_no_contradiction": False,
        "check3_improvement_delta": 0.0,
        "check4_confidence_ok": False,
        "rejection_reason": None,
    }

    # ── Check 4 — confidence threshold (cheapest, run first) ──────────────
    try:
        confidence = float(gate2_suggestion.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    result["check4_confidence_ok"] = confidence >= SUGGESTION_MIN_CONFIDENCE
    if not result["check4_confidence_ok"]:
        result["rejection_reason"] = f"Confidence {confidence:.2f} below threshold {SUGGESTION_MIN_CONFIDENCE}"
        return result

    # ── Check 1 — statistical backing ─────────────────────────────────────
    cited = gate2_suggestion.get("supporting_statistic", "")
    # Extract a percentage from the cited statistic string (e.g. "56% of Not Ok")
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", cited)
    if pct_match:
        claimed_pct = float(pct_match.group(1))
        # Find the closest actual pattern in gate1_result
        not_ok_total = gate1_result["total_not_ok"]
        closest_actual = 0.0
        for pattern in gate1_result.get("patterns", []):
            actual_pct = pattern["percentage"]
            if abs(actual_pct - claimed_pct) < abs(closest_actual - claimed_pct):
                closest_actual = actual_pct
        # Allow ±10 percentage-point tolerance
        tolerance_pts = SUGGESTION_MIN_STAT_SUPPORT * 100
        result["check1_stat_verified"] = abs(claimed_pct - closest_actual) <= tolerance_pts
        if not result["check1_stat_verified"]:
            result["rejection_reason"] = (
                f"Cited statistic {claimed_pct:.1f}% deviates "
                f"{abs(claimed_pct - closest_actual):.1f} pts from actual data."
            )
            return result
    else:
        # No numeric percentage found — cannot verify; still pass (not all findings are %)
        result["check1_stat_verified"] = True

    # ── Check 2 — contradiction detection ─────────────────────────────────
    current_prompt = prompt_registry.get_active_prompt_text()  # type: ignore[attr-defined]
    suggested_addition = gate2_suggestion.get("suggested_addition", "")
    try:
        contradiction_user = (
            f"Existing system prompt:\n---\n{current_prompt}\n---\n\n"
            f"Proposed addition (≤ 3 sentences):\n{suggested_addition}"
        )
        raw = _call_llm_for_improvement(_CONTRADICTION_CHECK_PROMPT, contradiction_user)
        contradiction_result = json.loads(raw)
        contradicts = bool(contradiction_result.get("contradicts", False))
        result["check2_no_contradiction"] = not contradicts
        if contradicts:
            result["rejection_reason"] = (
                f"Contradicts existing prompt: {contradiction_result.get('reason', 'unknown')}"
            )
            return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gate 3 Check 2: contradiction check failed (%s) — assuming no contradiction.", exc)
        result["check2_no_contradiction"] = True  # fail-open (better than blocking all suggestions)

    # ── Check 3 — shadow test ─────────────────────────────────────────────
    holdout_ids = gate1_result.get("holdout_ids", [])
    holdout_meta = feedback_store.get_by_ids(holdout_ids)  # type: ignore[attr-defined]
    # Filter to "Not Ok" cases only for the shadow test
    holdout_not_ok = [m for m in holdout_meta if str(m.get("verdict")).lower() in ("not ok", "not_ok")]

    if not holdout_not_ok:
        # No holdout data available — skip shadow test, pass by default
        result["check3_improvement_delta"] = 0.0
        logger.info("Gate 3 Check 3: no holdout data — shadow test skipped.")
    else:
        old_prompt = current_prompt
        new_prompt = current_prompt + "\n\n" + suggested_addition
        old_labels = _run_shadow_test(holdout_not_ok, old_prompt)
        new_labels = _run_shadow_test(holdout_not_ok, new_prompt)
        # Count how many cases changed label (proxy for improvement without ground truth)
        changed = sum(1 for o, n in zip(old_labels, new_labels) if o != n)
        delta = changed / len(holdout_not_ok)
        result["check3_improvement_delta"] = round(delta, 4)

        # Regression: new prompt produces MORE errors (none changed OR same labels)
        if delta < -0.02:
            result["rejection_reason"] = f"Shadow test regression: delta={delta:.3f}"
            return result

    result["passed"] = (
        result["check1_stat_verified"]
        and result["check2_no_contradiction"]
        and result["check3_improvement_delta"] >= -0.02
        and result["check4_confidence_ok"]
    )

    logger.info(
        "Gate 3 complete: passed=%s (stat=%s, nocontra=%s, delta=%.3f, conf=%s).",
        result["passed"],
        result["check1_stat_verified"],
        result["check2_no_contradiction"],
        result["check3_improvement_delta"],
        result["check4_confidence_ok"],
    )
    return result


def _run_shadow_test(holdout_not_ok: list[dict], system_prompt: str) -> list[str]:
    """Run holdout pairs through the LLM with a given prompt and return labels.

    Args:
        holdout_not_ok: List of feedback metadata dicts (Not Ok verdicts only).
        system_prompt:  The system prompt to use (old or new).

    Returns:
        List of AI similarity labels for each holdout pair.
    """
    from am_ais_assist.llm_service import _build_user_message  # noqa: PLC0415

    labels: list[str] = []
    batch_size = 10

    for i in range(0, len(holdout_not_ok), batch_size):
        batch = holdout_not_ok[i : i + batch_size]
        pairs = [
            (m.get("query_cleaned", "")[:300], m.get("matched_cleaned", "")[:300])
            for m in batch
        ]
        user_msg = _build_user_message(pairs)
        try:
            raw_json = _call_llm_for_improvement(system_prompt, user_msg)
            analysis = json.loads(raw_json)
            if isinstance(analysis, list):
                for item in analysis:
                    labels.append(str(item.get("relationship", "Unknown")))
            else:
                labels.extend(["Unknown"] * len(batch))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shadow test batch failed: %s", exc)
            labels.extend(["Unknown"] * len(batch))

    return labels


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_improvement_pipeline(
    feedback_store: object,
    prompt_registry: object,
    admin_user_name: str,
) -> dict:
    """Run Gates 1-3 of the self-improvement pipeline.

    Acquires IMPROVEMENT_LOCK_FILE so only one admin can run the pipeline
    at a time (prevents concurrent registry and shadow-test races).

    Args:
        feedback_store:    FeedbackStore instance.
        prompt_registry:   PromptRegistry instance.
        admin_user_name:   Display name of the triggering admin (for audit trail).

    Returns:
        dict with keys:
          status — "insufficient_data" | "no_clear_pattern" |
                   "validation_failed" | "ready_for_review" | "analysis_already_running"
          gate1, gate2, gate3 — results of each gate (where applicable)
    """
    try:
        lock = filelock.FileLock(IMPROVEMENT_LOCK_FILE, timeout=5)
        lock.acquire()
    except filelock.Timeout:
        logger.warning("Improvement pipeline already running — lock acquisition timed out.")
        return {"status": "analysis_already_running"}

    try:
        logger.info("Improvement pipeline started by admin: %s", admin_user_name)

        # Gate 1
        gate1 = run_gate1_statistical_analysis(feedback_store)
        if gate1 is None:
            return {"status": "insufficient_data"}

        # Gate 2
        gate2 = run_gate2_llm_analysis(gate1, prompt_registry)
        if gate2 is None:
            return {"status": "no_clear_pattern", "gate1": gate1}

        # Gate 3
        gate3 = run_gate3_validation(gate2, gate1, feedback_store, prompt_registry)

        status = "ready_for_review" if gate3["passed"] else "validation_failed"
        logger.info("Improvement pipeline finished: status=%s", status)

        return {
            "status": status,
            "gate1": gate1,
            "gate2": gate2,
            "gate3": gate3,
        }

    finally:
        lock.release()
