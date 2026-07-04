import os
import uuid
from datetime import datetime, timedelta

# Project root directory (3 levels up from src/am_ais_assist/config.py)
# NOTE: Only reliable when running from source. For installed packages the path
# resolves inside site-packages, which is why APP_DATA_DIR must be set in
# deployment environments (e.g. Docker).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Source directory (for prompts and source-related paths)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- File and Cache Paths ----
# APP_DATA_DIR env var lets Docker / CI override the data root without touching
# source code.  Falls back to <project_root>/data when running locally.
_default_data_dir = os.path.join(PROJECT_ROOT, "data")
DATA_DIR = os.getenv("APP_DATA_DIR", _default_data_dir)

CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
INPUT_DIR = os.path.join(DATA_DIR, "input")

# Prompt directory remains in src
PROMPT_DIR = os.path.join(SRC_DIR, "am_ais_assist", "prompts")

# Cache files
LLM_CACHE_FILE = os.path.join(
    CACHE_DIR, "llm_results_cache.json"
)  # Global fallback path; per-user path is preferred (see get_user_cache_dir())
CHROMA_PERSIST_DIR = os.path.join(CACHE_DIR, "base_index")
# NOTE: BASE_EMBEDDINGS_FILE and HASH_FILE removed (L-5) — stale FAISS-era constants no longer used.

# ---- Embedding Model Configuration ----#
EMBEDDING_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
EMBEDDING_MODEL = "nvidia/embeddings-nv-embed-qa-4"
EMBEDDING_API_VERSION = "2024-10-21"

# Prompt file for maintainability
SYSTEM_PROMPT_PATH = os.path.join(PROMPT_DIR, "system_prompt.txt")

# ---- Thresholds and Limits ----
# Similarity levels
DEFAULT_THRESHOLDS = {"exact": 1.0, "most": 0.9, "moderate": 0.7}

# ---- Hierarchical Section-Aware Comparison ----
SECTION_AUTO_MATCH_THRESHOLD = 0.90
SECTION_UNCERTAIN_THRESHOLD = 0.70
ITEM_NEW_REQ_THRESHOLD = 0.50
MAX_SAMPLE_ROWS_FOR_DETECTION = 10
SECTION_DETECTION_PROMPT_PATH = os.path.join(PROMPT_DIR, "section_detection_prompt.txt")
AGENT_DECISION_PROMPT_PATH = os.path.join(PROMPT_DIR, "agent_decision_prompt.txt")
PLACEHOLDER_NO_MATCH = "No equivalent found in base specification"
PLACEHOLDER_DELETED = "Not present in new specification"
PLACEHOLDER_DELETED_QUERY = "(Absent in new file — present in base)"

# File size limit (200 MB)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "200")) * 1024 * 1024

# Centralized configuration for scattered constants
MAX_TOKENS_FOR_TRUNCATION = 512
LLM_BATCH_TOKEN_LIMIT = 100000

# ---- Logging ----
LOG_LEVEL = "INFO"

# ---- LLM Configuration ----
LLM_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
LLM_MODEL = "meta/llama-3.1-70b-instruct"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
ENABLE_PROMPT_CACHING = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"

# LLM Optimization
LLM_ANALYSIS_MIN_THRESHOLD = 0.4
LLM_PERFECT_MATCH_THRESHOLD = 0.999

# Documentation URL — set this env var in deployment to the URL where mkdocs is served.
DOCS_URL: str = os.getenv("DOCS_URL", "")

# User session configuration
SESSION_TIMEOUT_HOURS = 24
# M-8: MAX_CONCURRENT_USERS is env-configurable; enforced in pipeline.py via semaphore
MAX_CONCURRENT_USERS = int(os.getenv("MAX_CONCURRENT_USERS", "20"))

# CONFIDENCE SCORE
HIGH_CONFIDENCE_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Phase 1 additions — Feedback Store
# ---------------------------------------------------------------------------

# ChromaDB collection for per-user feedback (prefix only — user_id appended at runtime)
# Full collection name pattern: "fb_{sanitized_user_id}"
FEEDBACK_CHROMA_DIR = os.path.join(CACHE_DIR, "feedback_index")
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "")
CHROMA_SERVER_PORT = os.getenv("CHROMA_SERVER_PORT", "8000")

# Global ChromaDB collection name — aggregates all user feedback for pattern analysis
FEEDBACK_GLOBAL_COLLECTION = "feedback_global"

# Minimum "Not OK" verdicts required before the self-improvement pipeline is unlocked.
# Protects against pattern analysis on statistically insignificant data.
FEEDBACK_MIN_NOT_OK = int(os.getenv("FEEDBACK_MIN_NOT_OK", "50"))

# Cosine similarity threshold for feedback recall.
# A stored feedback pair is considered "the same" as the current pair only when
# their embedding similarity meets or exceeds this value.
# 0.97 is deliberately tight — prevents false positives from superficially
# similar but semantically different requirement pairs.
FEEDBACK_SIMILARITY_THRESHOLD = float(os.getenv("FEEDBACK_SIMILARITY_THRESHOLD", "0.97"))

# Number of days after which feedback entries are eligible for cleanup.
# Aligns with the existing SESSION_TIMEOUT_HOURS pattern in cleanup_expired_sessions().
FEEDBACK_RETENTION_DAYS = int(os.getenv("FEEDBACK_RETENTION_DAYS", "90"))

# ---------------------------------------------------------------------------
# Phase 1 additions — Prompt Registry
# ---------------------------------------------------------------------------

# Directory that holds versioned prompt files and the registry JSON.
# Kept inside PROMPT_DIR so it travels with the source code.
PROMPT_VERSIONS_DIR = os.path.join(PROMPT_DIR, "versions")

# The single JSON file that records all prompt versions, their status,
# approval history, and canary/rollout metrics.
PROMPT_REGISTRY_FILE = os.path.join(PROMPT_VERSIONS_DIR, "registry.json")

# filelock path for safe concurrent read/write of the registry.
PROMPT_REGISTRY_LOCK_FILE = PROMPT_REGISTRY_FILE + ".lock"

# Percentage of sessions that receive the canary prompt version.
# Determined deterministically per session: hash(session_id) % 100 < CANARY_PERCENTAGE
# so the same user always gets the same version within a run.
CANARY_PERCENTAGE = int(os.getenv("CANARY_PERCENTAGE", "10"))

# Minimum number of canary sessions before a promote/rollback decision is made.
# Prevents premature decisions from a handful of sessions.
CANARY_MIN_SESSIONS = int(os.getenv("CANARY_MIN_SESSIONS", "20"))

# Minimum improvement in agreement rate (fraction) required to promote a canary
# to active. Example: 0.03 means the canary must be 3 percentage points better.
CANARY_MIN_IMPROVEMENT = float(os.getenv("CANARY_MIN_IMPROVEMENT", "0.03"))

# Fraction of global feedback held out for shadow testing (Gate 3 Check 3).
# These records are excluded from pattern analysis (Gate 1) and used only
# to measure whether a suggested prompt change actually improves classification.
SHADOW_TEST_HELD_OUT_PCT = float(os.getenv("SHADOW_TEST_HELD_OUT_PCT", "0.20"))

# Comma-separated list of usernames that can access the admin panel.
# Matched against the X-Auth-Request-Preferred-Username header value.
# Example env var:  AIS_ADMIN_USERS=john.doe,jane.smith
_raw_admins = os.getenv("AIS_ADMIN_USERS", "")
ADMIN_USERS: list[str] = [u.strip() for u in _raw_admins.split(",") if u.strip()]

# ---------------------------------------------------------------------------
# Phase 1 additions — Self-Improvement Pipeline
# ---------------------------------------------------------------------------

# Prompt files used exclusively by the self-improvement pipeline (Gates 2 & 3).
# Created in Phase 7 — paths defined here so all modules can import them.
PATTERN_ANALYSIS_PROMPT_PATH = os.path.join(PROMPT_DIR, "pattern_analysis_prompt.txt")
CONTRADICTION_CHECK_PROMPT_PATH = os.path.join(PROMPT_DIR, "contradiction_check_prompt.txt")

# Gate 2: reject a suggestion if the LLM's own confidence is below this value.
# Prevents low-certainty hallucinations from reaching the human review stage.
SUGGESTION_MIN_CONFIDENCE = float(os.getenv("SUGGESTION_MIN_CONFIDENCE", "0.60"))

# Gate 3 Check 1: a cited statistic is considered hallucinated if the LLM's
# claimed percentage deviates from the actual data by more than this fraction.
# Example: 0.10 means ±10 percentage-point tolerance.
SUGGESTION_MIN_STAT_SUPPORT = float(os.getenv("SUGGESTION_MIN_STAT_SUPPORT", "0.20"))

# filelock path that serialises concurrent admin-triggered improvement runs.
# Only one admin can run the 5-gate pipeline at a time to avoid race conditions
# on the registry and shadow-test data.
IMPROVEMENT_LOCK_FILE = os.path.join(CACHE_DIR, "improvement.lock")


# ---------------------------------------------------------------------------
# Helper functions (unchanged from original + new is_admin_user)
# ---------------------------------------------------------------------------


def generate_user_session() -> str:
    """Generate a unique session ID for each user."""
    return str(uuid.uuid4())


def get_user_cache_dir(user_session_id: str) -> str:
    """Get user-specific cache directory."""
    user_cache_dir = os.path.join(CACHE_DIR, f"user_{user_session_id}")
    os.makedirs(user_cache_dir, exist_ok=True)
    return user_cache_dir


def get_user_output_dir(user_session_id: str) -> str:
    """Get user-specific output directory."""
    user_output_dir = os.path.join(OUTPUT_DIR, f"user_{user_session_id}")
    os.makedirs(user_output_dir, exist_ok=True)
    return user_output_dir


def is_admin_user(user_name: str) -> bool:
    """Return True when user_name is in the configured ADMIN_USERS list.

    Comparison is case-insensitive and strips whitespace from both sides.
    Returns False when ADMIN_USERS is empty (no admins configured) so the
    admin panel is completely hidden in default deployments.

    Args:
        user_name: The username string from the auth header
                   (X-Auth-Request-Preferred-Username).

    Returns:
        bool — True if user_name matches any entry in ADMIN_USERS.
    """
    if not ADMIN_USERS:
        return False
    normalised = user_name.strip().lower()
    return normalised in {a.lower() for a in ADMIN_USERS}


def cleanup_expired_sessions() -> None:
    """Clean up expired user sessions and old feedback entries."""
    if not os.path.exists(CACHE_DIR):
        return

    cutoff_time = datetime.now() - timedelta(hours=SESSION_TIMEOUT_HOURS)

    for item in os.listdir(CACHE_DIR):
        if item.startswith("user_"):
            user_dir = os.path.join(CACHE_DIR, item)
            if os.path.isdir(user_dir):
                dir_time = datetime.fromtimestamp(os.path.getctime(user_dir))
                if dir_time < cutoff_time:
                    import shutil
                    shutil.rmtree(user_dir)
                    print(f"Cleaned up expired session: {item}")


def get_active_user_count() -> int:
    """Get number of active user sessions."""
    if not os.path.exists(CACHE_DIR):
        return 0
    return len([d for d in os.listdir(CACHE_DIR) if d.startswith("user_")])


def ensure_directories() -> None:
    """Create all required data directories on first run.

    Called once at application startup to guarantee that every path
    referenced by this config module exists before any module tries
    to write to it.  Safe to call multiple times (exist_ok=True).
    """
    dirs = [
        CACHE_DIR,
        OUTPUT_DIR,
        INPUT_DIR,
        CHROMA_PERSIST_DIR,
        FEEDBACK_CHROMA_DIR,
        PROMPT_VERSIONS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
