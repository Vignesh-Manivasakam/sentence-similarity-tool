"""
prompt_registry.py — Versioned prompt management with canary deployment support.

Architecture:
  PROMPT_VERSIONS_DIR/
    registry.json           ← single source of truth for all version metadata
    registry.json.lock      ← filelock for safe concurrent access
    system_prompt_v1.txt    ← initial version (copied from prompts/system_prompt.txt)
    system_prompt_v2.txt    ← created when admin approves a suggestion
    ...

Registry JSON schema (see _empty_registry() for full structure):
  {
    "active_version": "v1",
    "canary_version": null,
    "versions": {
      "v1": { ... full version metadata ... }
    }
  }

Canary assignment:
  Deterministic per session: hash(session_id) % 100 < CANARY_PERCENTAGE.
  The same session always receives the same prompt version within a run.
  Different sessions are distributed randomly but reproducibly.

Thread / multi-process safety:
  ALL reads and writes use filelock(PROMPT_REGISTRY_LOCK_FILE, timeout=15).
  In-memory prompt text is cached per version after the first disk read to
  avoid repeated I/O on every pipeline call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import filelock

from am_ais_assist.config import (
    CANARY_MIN_IMPROVEMENT,
    CANARY_MIN_SESSIONS,
    CANARY_PERCENTAGE,
    PROMPT_REGISTRY_FILE,
    PROMPT_REGISTRY_LOCK_FILE,
    PROMPT_VERSIONS_DIR,
    SYSTEM_PROMPT_PATH,
)

logger = logging.getLogger(__name__)

# In-memory cache: version_str → prompt_text (avoids repeated disk reads)
_prompt_text_cache: dict[str, str] = {}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _empty_registry() -> dict:
    return {
        "active_version": "v1",
        "canary_version": None,
        "versions": {},
    }


def _version_template(
    version: str,
    filename: str,
    finding: str,
    approved_by: str,
    supporting_statistic: str | None = None,
    shadow_test_improvement: float | None = None,
    status: str = "active",
) -> dict:
    return {
        "version": version,
        "filename": filename,
        "created_at": _now_iso(),
        "created_by": approved_by,
        "finding": finding,
        "supporting_statistic": supporting_statistic,
        "shadow_test_improvement": shadow_test_improvement,
        "approved_by": approved_by,
        "approved_at": _now_iso(),
        "status": status,
        "canary_start_at": None,
        "canary_sessions_count": 0,
        "canary_agreement_rate": None,
        "full_rollout_at": None,
        "retired_at": None,
        "retire_reason": None,
    }


class PromptRegistry:
    """Manages versioned system prompts with canary deployment.

    Designed as a process-level singleton — instantiate once via
    ``@st.cache_resource`` in ``app.py``.

    All public methods acquire PROMPT_REGISTRY_LOCK_FILE before touching
    the registry JSON or prompt files on disk.
    """

    def __init__(self) -> None:
        os.makedirs(PROMPT_VERSIONS_DIR, exist_ok=True)
        self._ensure_initial_version()
        logger.info("PromptRegistry initialised. Active: %s", self.get_active_version())

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict:
        """Read registry.json from disk. Must be called inside a filelock."""
        if not os.path.exists(PROMPT_REGISTRY_FILE):
            return _empty_registry()
        try:
            with open(PROMPT_REGISTRY_FILE, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("PromptRegistry: failed to read registry: %s", exc)
            return _empty_registry()

    def _save_registry(self, registry: dict) -> None:
        """Write registry.json to disk. Must be called inside a filelock."""
        try:
            with open(PROMPT_REGISTRY_FILE, "w", encoding="utf-8") as fh:
                json.dump(registry, fh, indent=2)
        except OSError as exc:
            logger.error("PromptRegistry: failed to write registry: %s", exc)

    def _read_prompt_file(self, filename: str) -> str:
        """Load a versioned prompt file from PROMPT_VERSIONS_DIR."""
        if filename in _prompt_text_cache:
            return _prompt_text_cache[filename]
        path = os.path.join(PROMPT_VERSIONS_DIR, filename)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
            _prompt_text_cache[filename] = text
            return text
        except FileNotFoundError:
            logger.error("PromptRegistry: prompt file not found: %s", path)
            return ""

    def _write_prompt_file(self, filename: str, text: str) -> None:
        """Write a new versioned prompt file to PROMPT_VERSIONS_DIR."""
        path = os.path.join(PROMPT_VERSIONS_DIR, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        _prompt_text_cache[filename] = text.strip()
        logger.info("PromptRegistry: wrote %s (%d chars).", filename, len(text))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_initial_version(self) -> None:
        """Create v1 from the current system_prompt.txt if not already done."""
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()
            if "v1" in registry.get("versions", {}):
                return  # already initialised

            # Read the current (hand-crafted) system prompt
            try:
                with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as fh:
                    original_prompt = fh.read()
            except FileNotFoundError:
                original_prompt = "You are a requirements analyst. Return results as a JSON array."
                logger.warning(
                    "PromptRegistry: system_prompt.txt not found — using inline default for v1."
                )

            # Copy it to versions/system_prompt_v1.txt
            v1_filename = "system_prompt_v1.txt"
            self._write_prompt_file(v1_filename, original_prompt)

            # Bootstrap registry
            registry["active_version"] = "v1"
            registry["canary_version"] = None
            registry["versions"]["v1"] = _version_template(
                version="v1",
                filename=v1_filename,
                finding="Initial version — original hand-crafted system prompt.",
                approved_by="system",
                status="active",
            )
            self._save_registry(registry)
            logger.info("PromptRegistry: v1 initialised from %s.", SYSTEM_PROMPT_PATH)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_active_version(self, session_id: str | None = None) -> str:
        """Return the prompt version applicable to this session.

        When a canary version is active, 10% of sessions (determined
        deterministically by ``hash(session_id) % 100``) receive the
        canary.  All other sessions receive the active (stable) version.

        Args:
            session_id: Streamlit session UUID (may be None).

        Returns:
            Version string such as ``"v1"`` or ``"v2"``.
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

        canary = registry.get("canary_version")
        if canary and session_id:
            bucket = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % 100
            if bucket < CANARY_PERCENTAGE:
                return canary

        return registry.get("active_version", "v1")

    def get_active_prompt_text(self, session_id: str | None = None) -> str:
        """Return the full prompt text applicable to this session.

        Args:
            session_id: Passed to ``get_active_version()`` for canary routing.

        Returns:
            The complete system prompt string.  Falls back to an inline
            default if the prompt file cannot be read.
        """
        version = self.get_active_version(session_id)
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

        version_meta = registry.get("versions", {}).get(version, {})
        filename = version_meta.get("filename", "")
        if not filename:
            logger.warning("PromptRegistry: no filename for version %s.", version)
            return ""

        return self._read_prompt_file(filename)

    def get_all_versions(self) -> list[dict]:
        """Return all version metadata sorted newest-first.

        Used by the admin panel to render the version history table.
        Includes a ``prompt_preview`` field (first 200 chars of the prompt).
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

        versions = list(registry.get("versions", {}).values())
        # Sort by created_at descending (newest first)
        versions.sort(key=lambda v: v.get("created_at", ""), reverse=True)

        # Attach prompt preview (read outside lock — files are immutable once written)
        for v in versions:
            filename = v.get("filename", "")
            if filename:
                text = self._read_prompt_file(filename)
                v["prompt_preview"] = text[:200] + ("..." if len(text) > 200 else "")
            else:
                v["prompt_preview"] = ""

        return versions

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def create_new_version(
        self,
        prompt_text: str,
        finding: str,
        approved_by: str,
        supporting_statistic: str | None = None,
        shadow_test_improvement: float | None = None,
    ) -> str:
        """Create a new canary version from an approved prompt suggestion.

        The new version starts in ``"canary"`` status.  It is deployed to
        ``CANARY_PERCENTAGE`` % of sessions immediately.  Full promotion
        requires either an admin action or the canary monitor confirming
        sufficient improvement.

        Args:
            prompt_text:             Full text of the new prompt.
            finding:                 Human-readable description of the change.
            approved_by:             Username of the approving admin.
            supporting_statistic:    Statistic that backed this change.
            shadow_test_improvement: Improvement delta from Gate 3 shadow test.

        Returns:
            New version string (e.g. ``"v2"``).
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

            # Determine next version number
            existing = [int(v[1:]) for v in registry.get("versions", {}) if v.startswith("v")]
            next_num = max(existing, default=0) + 1
            new_ver = f"v{next_num}"
            filename = f"system_prompt_{new_ver}.txt"

            self._write_prompt_file(filename, prompt_text)

            registry["versions"][new_ver] = _version_template(
                version=new_ver,
                filename=filename,
                finding=finding,
                approved_by=approved_by,
                supporting_statistic=supporting_statistic,
                shadow_test_improvement=shadow_test_improvement,
                status="canary",
            )
            registry["versions"][new_ver]["canary_start_at"] = _now_iso()
            registry["canary_version"] = new_ver

            self._save_registry(registry)

        logger.info(
            "PromptRegistry: canary version %s created (approved_by=%s).", new_ver, approved_by
        )
        return new_ver

    def promote_canary_to_active(self) -> bool:
        """Promote the current canary to active and retire the old active version.

        Returns:
            True on success, False when no canary is active.
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()
            canary_ver = registry.get("canary_version")
            if not canary_ver:
                logger.warning("PromptRegistry: promote called but no canary active.")
                return False

            old_active = registry["active_version"]

            # Retire old active
            registry["versions"][old_active]["status"] = "retired"
            registry["versions"][old_active]["retired_at"] = _now_iso()
            registry["versions"][old_active]["retire_reason"] = f"Superseded by {canary_ver}"

            # Promote canary
            registry["versions"][canary_ver]["status"] = "active"
            registry["versions"][canary_ver]["full_rollout_at"] = _now_iso()
            registry["active_version"] = canary_ver
            registry["canary_version"] = None

            self._save_registry(registry)

        # Invalidate in-memory cache for the old active version (now retired)
        logger.info(
            "PromptRegistry: promoted %s to active, retired %s.", canary_ver, old_active
        )
        return True

    def rollback_canary(self) -> bool:
        """Discard the canary version (reject it) without touching the active version.

        Returns:
            True on success, False when no canary is active.
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()
            canary_ver = registry.get("canary_version")
            if not canary_ver:
                return False

            registry["versions"][canary_ver]["status"] = "rejected"
            registry["versions"][canary_ver]["retired_at"] = _now_iso()
            registry["versions"][canary_ver]["retire_reason"] = "Rolled back — insufficient improvement"
            registry["canary_version"] = None

            self._save_registry(registry)

        logger.info("PromptRegistry: canary %s rolled back.", canary_ver)
        return True

    def rollback_to_version(self, target_version: str) -> bool:
        """Emergency rollback: set an arbitrary previous version as active.

        The current active version is retired.  Any active canary is also
        rolled back before the target is promoted.

        Args:
            target_version: Version string to restore (e.g. ``"v1"``).

        Returns:
            True on success, False when target_version does not exist.
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

            if target_version not in registry.get("versions", {}):
                logger.error("PromptRegistry: rollback target %s not found.", target_version)
                return False

            # Roll back canary first if active
            canary_ver = registry.get("canary_version")
            if canary_ver:
                registry["versions"][canary_ver]["status"] = "rejected"
                registry["versions"][canary_ver]["retired_at"] = _now_iso()
                registry["versions"][canary_ver]["retire_reason"] = "Superseded by emergency rollback"
                registry["canary_version"] = None

            # Retire current active
            old_active = registry["active_version"]
            if old_active != target_version:
                registry["versions"][old_active]["status"] = "retired"
                registry["versions"][old_active]["retired_at"] = _now_iso()
                registry["versions"][old_active]["retire_reason"] = (
                    f"Emergency rollback to {target_version}"
                )

            # Restore target
            registry["versions"][target_version]["status"] = "active"
            registry["versions"][target_version]["full_rollout_at"] = _now_iso()
            registry["active_version"] = target_version

            self._save_registry(registry)

        logger.warning(
            "PromptRegistry: emergency rollback to %s complete.", target_version
        )
        return True

    def record_canary_session(
        self, canary_version: str, agreement_rate: float
    ) -> dict:
        """Update the canary's session count and rolling average agreement rate.

        Called from ``app.py`` after a user saves reviews in a canary session.
        Returns current canary stats for the caller to decide on promotion.

        Args:
            canary_version:  The canary version string (from session_state).
            agreement_rate:  Fraction of reviewed rows marked "OK" in this session.

        Returns:
            dict with keys: sessions_count, canary_agreement_rate, decision
            where decision is one of: "pending", "promote", "rollback", "inconclusive"
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

            if canary_version not in registry.get("versions", {}):
                return {"decision": "pending"}

            ver = registry["versions"][canary_version]
            prev_count = ver.get("canary_sessions_count", 0)
            prev_rate = ver.get("canary_agreement_rate") or 0.0

            # Incremental average: weighted mean of all sessions seen so far
            new_count = prev_count + 1
            new_rate = (prev_rate * prev_count + agreement_rate) / new_count

            ver["canary_sessions_count"] = new_count
            ver["canary_agreement_rate"] = round(new_rate, 4)

            self._save_registry(registry)

        # Determine decision (outside lock — read-only)
        decision = "pending"
        if new_count >= CANARY_MIN_SESSIONS:
            # Get active version agreement rate for comparison
            with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
                registry2 = self._load_registry()
            active_ver = registry2["active_version"]
            active_meta = registry2["versions"].get(active_ver, {})
            baseline_rate = active_meta.get("canary_agreement_rate") or 0.70  # fallback baseline

            delta = new_rate - baseline_rate
            if delta >= CANARY_MIN_IMPROVEMENT:
                decision = "promote"
            elif delta < -0.02:
                decision = "rollback"
            else:
                decision = "inconclusive"

        logger.debug(
            "PromptRegistry: canary %s session recorded (count=%d, rate=%.3f, decision=%s).",
            canary_version,
            new_count,
            new_rate,
            decision,
        )
        return {
            "sessions_count": new_count,
            "canary_agreement_rate": round(new_rate, 4),
            "decision": decision,
        }

    def get_canary_status(self) -> dict:
        """Return current canary monitoring snapshot.

        Returns:
            dict with status ("no_canary", "pending", "promote", "rollback",
            "inconclusive"), plus counts and rates when a canary is active.
        """
        with filelock.FileLock(PROMPT_REGISTRY_LOCK_FILE, timeout=15):
            registry = self._load_registry()

        canary_ver = registry.get("canary_version")
        if not canary_ver:
            return {"status": "no_canary"}

        ver = registry["versions"].get(canary_ver, {})
        sessions = ver.get("canary_sessions_count", 0)
        canary_rate = ver.get("canary_agreement_rate") or 0.0

        active_ver = registry["active_version"]
        active_meta = registry["versions"].get(active_ver, {})
        baseline_rate = active_meta.get("canary_agreement_rate") or 0.70

        decision = "pending"
        if sessions >= CANARY_MIN_SESSIONS:
            delta = canary_rate - baseline_rate
            if delta >= CANARY_MIN_IMPROVEMENT:
                decision = "promote"
            elif delta < -0.02:
                decision = "rollback"
            else:
                decision = "inconclusive"

        return {
            "status": decision,
            "canary_version": canary_ver,
            "active_version": active_ver,
            "sessions_count": sessions,
            "canary_agreement_rate": canary_rate,
            "baseline_agreement_rate": baseline_rate,
            "delta": round(canary_rate - baseline_rate, 4),
        }
