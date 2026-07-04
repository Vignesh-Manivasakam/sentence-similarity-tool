import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Logging — configured before any other app imports so all modules
# inherit the same handlers from the root logger.
# ---------------------------------------------------------------------------
for _handler in logging.root.handlers[:]:
    logging.root.removeHandler(_handler)

# Ensure local `src` package directory is on `sys.path` so local imports work
# when running the app from the repository (e.g., `poe run-frontend`).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_path = os.path.join(_repo_root, "src")
if os.path.isdir(_src_path) and _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from am_ais_assist.config import LOG_LEVEL

_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "app.log")

_file_handler = RotatingFileHandler(
    _log_file, mode="a", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.getLogger("").setLevel(LOG_LEVEL.upper())
logging.getLogger("").addHandler(_file_handler)
logging.getLogger("").addHandler(_console_handler)
logging.info("Logging configured successfully.")

# ---------------------------------------------------------------------------
# Streamlit page config — must be the first st.* call
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AIS")

# ---------------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------------
from am_ais_assist.config import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    DOCS_URL,
    LLM_ANALYSIS_MIN_THRESHOLD,
    LLM_PERFECT_MATCH_THRESHOLD,
    MAX_FILE_SIZE,
    ensure_directories,
    is_admin_user,
)
from am_ais_assist.pipeline import run_similarity_pipeline
from am_ais_assist.postprocess import create_highlighted_excel, df_to_html_table, display_summary
from am_ais_assist.progress_manager import UnifiedProgressManager
from am_ais_assist.utils import plot_embeddings

# Ensure all required directories exist on startup
ensure_directories()


# ---------------------------------------------------------------------------
# Global singletons — initialised once, shared across all user sessions
# ---------------------------------------------------------------------------
@st.cache_resource
def get_global_cache_manager():
    from am_ais_assist.cache_manager import GlobalCacheManager
    return GlobalCacheManager()


@st.cache_resource
def get_feedback_store():
    """Process-level FeedbackStore singleton."""
    from am_ais_assist.feedback_store import FeedbackStore
    return FeedbackStore()


@st.cache_resource
def get_prompt_registry():
    """Process-level PromptRegistry singleton."""
    from am_ais_assist.prompt_registry import PromptRegistry
    return PromptRegistry()


cache_manager = get_global_cache_manager()
feedback_store = get_feedback_store()
prompt_registry = get_prompt_registry()

# Declare the custom interactive Excel-like HTML/JS report viewer component
_component_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "am_ais_assist", "report_viewer")
_report_viewer = components.declare_component("report_viewer", path=_component_path)


def start_periodic_cleanup():
    """Start a background thread that periodically cleans up expired cache files (Risk 4 fix)."""
    import threading
    import time
    def _cleanup_loop():
        # Wait 10 minutes on startup, then clean up every 2 hours
        time.sleep(600)
        while True:
            try:
                from am_ais_assist.config import cleanup_expired_sessions
                cleanup_expired_sessions()
                logging.info("Background cleanup: expired sessions purged.")
            except Exception as e:
                logging.warning("Background cleanup failed: %s", e)
            # Sleep for 2 hours
            time.sleep(7200)

    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# M-3: Startup cleanup — run once per process to purge expired session dirs
# ---------------------------------------------------------------------------
@st.cache_resource
def _run_startup_cleanup():
    try:
        from am_ais_assist.config import cleanup_expired_sessions
        cleanup_expired_sessions()
        logging.info("Startup: expired session cleanup complete.")
        # Start background periodic cache cleanup thread
        start_periodic_cleanup()
        logging.info("Startup: background periodic cache cleanup thread started.")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Startup cleanup failed (non-fatal): %s", exc)


_run_startup_cleanup()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def initialize_user_session() -> str:
    """Create a unique session ID and capture user info for this browser session."""
    if "user_session_id" not in st.session_state:
        from am_ais_assist.config import generate_user_session
        st.session_state.user_session_id = generate_user_session()
        logging.info("New user session created: %s", st.session_state.user_session_id)

    if "user_name" not in st.session_state:
        # Check gateway secret if configured in env (Risk 1 fix — header spoofing protection)
        gateway_secret = os.getenv("GATEWAY_SECRET", "")
        if gateway_secret:
            req_secret = st.context.headers.get("X-Gateway-Secret", "")
            if req_secret != gateway_secret:
                logging.warning("Security Warning: request headers missing or invalid X-Gateway-Secret. Reverting to guest.")
                st.session_state.user_name = "guest_user"
                st.session_state.user_email = "guest_user@bosch.com"
                st.session_state.user_id = "guest_user_pseudonym"
                return st.session_state.user_session_id

        st.session_state.user_name = st.context.headers.get(
            "X-Auth-Request-Preferred-Username", "local_user"
        )
        st.session_state.user_email = st.context.headers.get(
            "X-Auth-Request-Email", "local_mail@bosch.com"
        )
        st.session_state.user_id = st.context.headers.get(
            "X-Auth-Request-User", "local_user_pseudonym"
        )
        logging.info(
            "User identified: %s (session: %s)",
            st.session_state.user_name,
            st.session_state.user_session_id,
        )

    return st.session_state.user_session_id


def initialize_file_session() -> None:
    """Initialise all file-related session state keys to their defaults."""
    defaults = {
        "files_processed": False,
        "base_file_data": None,
        "check_file_data": None,
        "base_columns": [],
        "check_columns": [],
        "last_base_file_name": None,
        "last_check_file_name": None,
        "base_embeddings": None,
        "user_embeddings": None,
        "base_data": None,
        "user_data": None,
        "stats": (0, 0),
        "results_df": None,
        "total_tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        # Phase 5 — session persistence
        "base_file_hash": "",
        "check_file_hash": "",
        "active_prompt_version": "v1",
        # Phase 6 — review panel
        "review_statuses": {},   # {query_id: "Pending" | "OK" | "Not OK"}
        "review_remarks": {},    # {query_id: user_remark_text}
        "reviews_saved": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Phase 6 — Filter Panel
# ---------------------------------------------------------------------------


def render_filter_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter controls and return a filtered copy of df.

    Adds a Review_Status column from session_state.review_statuses before
    filtering so the Review filter works correctly.
    """
    # Attach review statuses to the df view (not stored in the df itself yet)
    df_view = df.copy()
    df_view["Review_Status"] = df_view["Query_Object_Identifier"].map(
        lambda qid: st.session_state.review_statuses.get(str(qid), "Pending")
    )

    with st.expander("🔍 Filter Results", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            level_options = sorted(df_view["Similarity_Level"].dropna().unique().tolist()) \
                if "Similarity_Level" in df_view.columns else []
            selected_levels = st.multiselect(
                "Similarity Level", ["All"] + level_options, default=["All"],
                key="filter_level"
            )

        with col2:
            score_range = st.slider(
                "Score Range", 0.0, 1.0, (0.0, 1.0), step=0.05, key="filter_score"
            )

        with col3:
            selected_review = st.selectbox(
                "Review Status", ["All", "Pending", "OK", "Not OK"], key="filter_review"
            )

        with col4:
            # Derive section prefix from Query_Object_Identifier (first 2 parts)
            if "Query_Object_Identifier" in df_view.columns:
                def _section(qid: str) -> str:
                    parts = str(qid).split(".")
                    return ".".join(parts[:2]) if len(parts) >= 2 else str(qid)
                section_options = sorted(df_view["Query_Object_Identifier"].map(_section).unique())
                selected_section = st.selectbox(
                    "Section", ["All"] + list(section_options), key="filter_section"
                )
            else:
                selected_section = "All"

    # Apply filters
    filtered = df_view.copy()

    if "All" not in selected_levels and selected_levels:
        filtered = filtered[filtered["Similarity_Level"].isin(selected_levels)]

    # Score filter: keep rows where score is in range OR score is non-numeric
    if "Similarity_Score" in filtered.columns:
        numeric_scores = pd.to_numeric(filtered["Similarity_Score"], errors="coerce")
        score_mask = (
            (numeric_scores >= score_range[0]) & (numeric_scores <= score_range[1])
        ) | numeric_scores.isna()
        filtered = filtered[score_mask]

    if selected_review != "All":
        filtered = filtered[filtered["Review_Status"] == selected_review]

    if selected_section != "All" and "Query_Object_Identifier" in filtered.columns:
        filtered = filtered[
            filtered["Query_Object_Identifier"].map(_section) == selected_section
        ]
    return filtered


# ---------------------------------------------------------------------------
# Phase 6 — Admin Panel
# ---------------------------------------------------------------------------


def render_admin_panel() -> None:
    """Admin-only panel for prompt management and self-improvement pipeline."""
    user_name = st.session_state.get("user_name", "")
    if not is_admin_user(user_name):
        return  # silently hidden from non-admins

    with st.expander("⚙️ Admin: Prompt Management", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📊 Feedback Stats", "🧠 Improve Prompt", "📜 Version History"])

        with tab1:
            _render_admin_feedback_stats()

        with tab2:
            _render_admin_improve_prompt()

        with tab3:
            _render_admin_version_history()


def _render_admin_feedback_stats() -> None:
    stats = feedback_store.get_feedback_statistics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", stats["total_reviews"])
    col2.metric("Not OK Verdicts", stats["total_not_ok"])
    col3.metric(
        "Analysis Ready",
        "✅ Yes" if stats["min_threshold_met"] else "❌ No",
        help=f"Requires ≥ {__import__('am_ais_assist.config', fromlist=['FEEDBACK_MIN_NOT_OK']).FEEDBACK_MIN_NOT_OK} Not OK verdicts",
    )

    if stats["by_level"]:
        import plotly.express as px
        level_df = pd.DataFrame(
            [{"Level": k, "Count": v} for k, v in stats["by_level"].items()]
        )
        fig = px.bar(level_df, x="Level", y="Count", title="Not OK Verdicts by AI Level", text="Count")
        fig.update_layout(paper_bgcolor="#1e1e2f", plot_bgcolor="#2a2a3c", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)


def _render_admin_improve_prompt() -> None:
    canary_status = prompt_registry.get_canary_status()

    if canary_status["status"] == "no_canary":
        st.info(
            f"Active prompt: **{prompt_registry.get_active_version()}**. "
            f"No canary running."
        )
        if st.button("🔬 Run Pattern Analysis (Gates 1–3)", key="btn_run_analysis"):
            with st.spinner("Analysing feedback patterns…"):
                try:
                    from am_ais_assist.self_improve import run_improvement_pipeline  # noqa: PLC0415
                    result = run_improvement_pipeline(
                        feedback_store,
                        prompt_registry,
                        admin_user_name=st.session_state.get("user_name", "admin"),
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    return

            status = result.get("status", "unknown")
            if status == "insufficient_data":
                st.warning("Not enough feedback data yet. Collect more reviews and try again.")
            elif status == "no_clear_pattern":
                st.info("No clear improvement pattern found in current feedback.")
            elif status == "validation_failed":
                gate3 = result.get("gate3", {})
                st.error(f"Validation failed: {gate3.get('rejection_reason', 'Unknown reason')}")
            elif status == "analysis_already_running":
                st.warning("Analysis already running — please wait.")
            elif status == "ready_for_review":
                _render_suggestion_review(result)
            else:
                st.warning(f"Unexpected status: {status}")

    else:
        # Canary is active — show monitoring dashboard
        st.subheader(f"🐤 Canary Active: {canary_status['canary_version']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Canary Sessions", canary_status.get("sessions_count", 0))
        c2.metric(
            "Canary Agreement",
            f"{canary_status.get('canary_agreement_rate', 0):.1%}",
            delta=f"{canary_status.get('delta', 0):+.1%}",
        )
        c3.metric("Decision", canary_status.get("status", "pending").upper())

        col_promote, col_rollback = st.columns(2)
        with col_promote:
            if st.button("🚀 Promote Canary to Active", key="btn_promote"):
                if prompt_registry.promote_canary_to_active():
                    st.success("Canary promoted to active!")
                    st.rerun()
        with col_rollback:
            if st.button("⏪ Rollback Canary", key="btn_rollback"):
                if prompt_registry.rollback_canary():
                    st.info("Canary rolled back.")
                    st.rerun()


def _render_suggestion_review(result: dict) -> None:
    gate2 = result["gate2"]
    gate3 = result["gate3"]

    st.success("✅ All validation gates passed. Review the suggestion below:")

    c1, c2 = st.columns(2)
    c1.metric("Shadow Test Improvement", f"+{gate3['check3_improvement_delta'] * 100:.1f}%")
    c2.metric("LLM Confidence", f"{float(gate2.get('confidence', 0)) * 100:.0f}%")

    st.write(f"**Finding:** {gate2.get('finding', '')}")
    st.write(f"**Evidence:** {gate2.get('supporting_statistic', '')}")

    edited_addition = st.text_area(
        "Proposed addition to system prompt (edit if needed):",
        value=gate2.get("suggested_addition", ""),
        height=120,
        key="suggestion_text_area",
    )

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("✅ Approve & Create Canary Version", key="btn_approve"):
            current_prompt = prompt_registry.get_active_prompt_text()
            new_prompt = current_prompt + "\n\n" + edited_addition
            new_ver = prompt_registry.create_new_version(
                prompt_text=new_prompt,
                finding=gate2.get("finding", ""),
                approved_by=st.session_state.get("user_name", "admin"),
                supporting_statistic=gate2.get("supporting_statistic"),
                shadow_test_improvement=gate3.get("check3_improvement_delta"),
            )
            st.success(
                f"Canary version **{new_ver}** created! "
                f"~{__import__('am_ais_assist.config', fromlist=['CANARY_PERCENTAGE']).CANARY_PERCENTAGE}% "
                f"of sessions will receive the new prompt."
            )
            st.rerun()
    with col_reject:
        if st.button("❌ Reject Suggestion", key="btn_reject"):
            st.info("Suggestion rejected. Collect more feedback and try again.")


def _render_admin_version_history() -> None:
    versions = prompt_registry.get_all_versions()
    if not versions:
        st.info("No versions recorded yet.")
        return

    rows = []
    for v in versions:
        rows.append({
            "Version": v["version"],
            "Status": v.get("status", ""),
            "Created": str(v.get("created_at", ""))[:10],
            "Approved By": v.get("approved_by", ""),
            "Finding": str(v.get("finding", ""))[:60],
            "Shadow Test": (
                f"+{v['shadow_test_improvement'] * 100:.1f}%"
                if v.get("shadow_test_improvement") is not None
                else "—"
            ),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("⚠️ Emergency Rollback")
    retired_versions = [v["version"] for v in versions if v.get("status") == "retired"]
    if retired_versions:
        rollback_target = st.selectbox(
            "Rollback to version:", retired_versions, key="rollback_target"
        )
        if st.button("Execute Emergency Rollback", key="btn_emergency_rollback", type="primary"):
            if prompt_registry.rollback_to_version(rollback_target):
                st.warning(f"Emergency rollback to {rollback_target} complete.")
                st.rerun()
    else:
        st.info("No retired versions available for rollback.")


# ---------------------------------------------------------------------------
# Sidebar layout
# ---------------------------------------------------------------------------


def load_sidebar():  # noqa: PLR0915
    """Render file upload widgets, column selectors, and settings."""
    with st.sidebar:
        st.header("📂 File Selection & Settings")

        with st.expander("Upload Files", expanded=True):
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # ── Base file ────────────────────────────────────────────────
            base_file = st.file_uploader("Base Excel File 📑", type=["xlsx"])
            base_id_col, base_text_col, base_meta_cols = None, None, []

            if base_file:
                if base_file.size > MAX_FILE_SIZE:
                    st.error(f"Base file size exceeds {MAX_FILE_SIZE / 1024 / 1024:.0f} MB limit")
                    base_file = None
                else:
                    file_changed = st.session_state.last_base_file_name != base_file.name
                    if file_changed or st.session_state.base_file_data is None:
                        try:
                            with st.spinner("Reading base file columns..."):
                                st.session_state.base_file_data = pd.read_excel(base_file, nrows=0)
                                st.session_state.base_columns = list(
                                    st.session_state.base_file_data.columns
                                )
                                st.session_state.last_base_file_name = base_file.name
                                st.session_state.files_processed = True
                        except Exception as exc:
                            st.error(f"Error reading base file: {exc}")
                            base_file = None
                            st.session_state.base_file_data = None
                            st.session_state.base_columns = []

                    if st.session_state.base_file_data is not None:
                        st.markdown(
                            '<p class="upload-success">✅ Base file uploaded!</p>',
                            unsafe_allow_html=True,
                        )
                        base_id_col = st.selectbox(
                            "Select Base Identifier Column",
                            st.session_state.base_columns,
                            key="base_id_col",
                        )
                        base_text_col = st.selectbox(
                            "Select Base Text Column",
                            st.session_state.base_columns,
                            key="base_text_col",
                        )
                        remaining_cols = [
                            c
                            for c in st.session_state.base_columns
                            if c not in [base_id_col, base_text_col]
                        ]
                        base_meta_cols = st.multiselect(
                            "Select Base additional Columns (Optional)",
                            remaining_cols,
                            key="base_meta_cols",
                        )

            # ── Check file ───────────────────────────────────────────────
            check_file = st.file_uploader("Check Excel File 📝", type=["xlsx"])
            check_id_col, check_text_col, check_meta_cols = None, None, []

            if check_file:
                if check_file.size > MAX_FILE_SIZE:
                    st.error(f"Check file size exceeds {MAX_FILE_SIZE / 1024 / 1024:.0f} MB limit")
                    check_file = None
                else:
                    file_changed = st.session_state.last_check_file_name != check_file.name
                    if file_changed or st.session_state.check_file_data is None:
                        try:
                            with st.spinner("Reading check file columns..."):
                                st.session_state.check_file_data = pd.read_excel(check_file, nrows=0)
                                st.session_state.check_columns = list(
                                    st.session_state.check_file_data.columns
                                )
                                st.session_state.last_check_file_name = check_file.name
                                st.session_state.files_processed = True
                        except Exception as exc:
                            st.error(f"Error reading check file: {exc}")
                            check_file = None
                            st.session_state.check_file_data = None
                            st.session_state.check_columns = []

                    if st.session_state.check_file_data is not None:
                        st.markdown(
                            '<p class="upload-success">✅ Target file uploaded!</p>',
                            unsafe_allow_html=True,
                        )
                        check_id_col = st.selectbox(
                            "Select Target Identifier Column",
                            st.session_state.check_columns,
                            key="check_id_col",
                        )
                        check_text_col = st.selectbox(
                            "Select Target Text Column",
                            st.session_state.check_columns,
                            key="check_text_col",
                        )
                        remaining_cols = [
                            c
                            for c in st.session_state.check_columns
                            if c not in [check_id_col, check_text_col]
                        ]
                        check_meta_cols = st.multiselect(
                            "Select Target additional Columns (Optional)",
                            remaining_cols,
                            key="check_meta_cols",
                        )

            top_k = st.number_input("Top K Matches 🎯", min_value=1, max_value=10, value=3)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Relationship Classification Guide", expanded=False):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
                ✅ **Exact Match**: 1.0
                *Perfect string match*

                🟢 **Equivalent**: 0.95 – 1.0
                *Same meaning, different wording*

                🟡 **Related**: 0.50 – 0.94
                *Related concepts, partial overlap*

                🔴 **Contradictory**: 0.00 – 0.20
                *Opposite / conflicting meanings*

                🔵 **New Requirement**: —
                *Exists in new file only*

                🟣 **Deleted**: —
                *Exists in base file only*
            """)
            st.markdown("</div>", unsafe_allow_html=True)

        run_btn = st.button("🚀 Run Similarity Search")

    return (
        base_file, check_file, top_k, run_btn,
        base_id_col, base_text_col,
        check_id_col, check_text_col,
        base_meta_cols, check_meta_cols,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    """Entry point for the Streamlit application."""

    # ── Top-right Documentation & Help buttons ────────────────────────────
    btn_col_spacer, btn_col_doc, btn_col_help = st.columns([8, 1, 1])
    with btn_col_doc:
        st.link_button(
            "📖 Docs",
            "https://pages.github.boschdevcloud.com/modana/am-ais-assist/",
            use_container_width=True,
        )
    with btn_col_help:
        with st.popover("❓ Help", use_container_width=True):
            st.markdown("**Need assistance?**")
            st.markdown(
                '<a class="help-link-btn" '
                'href="mailto:Manivasakam.Vignesh@in.bosch.com">'
                "✉️ Email Support</a>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<a class="help-link-btn" '
                'href="https://teams.microsoft.com/l/channel/19%3APmXVfo23UBDPU8Z41yGuzSbh627VmG7g5OF-YGZLvyE1%40thread.tacv2/AIS%20Tenant%40MODANA-BPC?groupId=b1b89dc4-1672-4b18-85ca-323ba1d25f34&tenantId=0ae51e19-07c8-4e4b-bb6d-648ee58410f4&ngc=true" '
                'target="_blank">'
                "💬 Teams Channel</a>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<h1 style='text-align: center;'>📘 AIS — 🧠 AI Based Similarity Assist 🔍</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h6 style='text-align: center;'>For any Queries Contact: <br>Vignesh Manivasakam (MS/ENP42-VM)</h6>",
        unsafe_allow_html=True,
    )

    user_session_id = initialize_user_session()
    initialize_file_session()

    try:
        with open("scripts/css/custom.css") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

    (
        base_file, check_file, top_k, run_btn,
        base_id_col, base_text_col,
        check_id_col, check_text_col,
        base_meta_cols, check_meta_cols,
    ) = load_sidebar()

    # ── Run pipeline ─────────────────────────────────────────────────────
    if run_btn:
        if not base_file or not check_file:
            st.warning("⚠️ Please upload both Excel files.")
            return
        if not base_id_col or not base_text_col or not check_id_col or not check_text_col:
            st.warning("⚠️ Please select identifier and text columns for both files.")
            return
        if base_id_col == base_text_col:
            st.error("❌ **Base file error**: Identifier and Text columns must be different.")
            return
        if check_id_col == check_text_col:
            st.error("❌ **Check file error**: Identifier and Text columns must be different.")
            return

        unified_progress = st.progress(0, text="🚀 Initialising pipeline...")
        progress_manager = UnifiedProgressManager(unified_progress)

        with st.spinner("Running similarity search with LLM analysis..."):
            try:
                # Phase 5: pass feedback_store and prompt_registry
                (
                    results,
                    base_embeddings,
                    user_embeddings,
                    base_data,
                    user_data,
                    base_skipped_count,
                    user_skipped_count,
                    llm_tokens,
                    base_file_hash,         # Phase 5 — new
                    check_file_hash,        # Phase 5 — new
                    active_prompt_version,  # Phase 5 — new
                ) = run_similarity_pipeline(
                    base_file,
                    check_file,
                    top_k,
                    None,
                    progress_manager,
                    base_id_col,
                    base_text_col,
                    check_id_col,
                    check_text_col,
                    base_meta_cols,
                    check_meta_cols,
                    user_session_id,
                    cache_manager,
                    feedback_store=feedback_store,
                    prompt_registry=prompt_registry,
                    user_id=st.session_state.get("user_id", "local_user_pseudonym"),
                )

                st.session_state.results_df = pd.DataFrame(results)
                st.session_state.base_data = base_data
                st.session_state.user_data = user_data
                st.session_state.base_embeddings = base_embeddings
                st.session_state.user_embeddings = user_embeddings
                st.session_state.stats = (base_skipped_count, user_skipped_count)
                st.session_state.total_tokens = llm_tokens

                # Phase 5 — session persistence
                st.session_state.base_file_hash = base_file_hash
                st.session_state.check_file_hash = check_file_hash
                st.session_state.active_prompt_version = active_prompt_version

                # Phase 5 — pre-load prior reviews for this file combination
                prior_reviews = feedback_store.load_run_reviews(
                    user_id=st.session_state.get("user_id", user_session_id),
                    base_file_hash=base_file_hash,
                    check_file_hash=check_file_hash,
                )
                # Initialise review statuses: prior reviews override "Pending" default
                review_statuses: dict[str, str] = {}
                review_remarks: dict[str, str] = {}
                for r in results:
                    qid = str(r.get("Query_Object_Identifier", ""))
                    prior = prior_reviews.get(qid, {})
                    review_statuses[qid] = prior.get("verdict", "Pending")
                    review_remarks[qid] = prior.get("user_remark", "")
                st.session_state.review_statuses = review_statuses
                st.session_state.review_remarks = review_remarks
                st.session_state.reviews_saved = False

                unified_progress.progress(1.0, text="🎉 All phases complete!")
                if prior_reviews:
                    st.success(
                        f"✅ Similarity Search + LLM Analysis Complete! "
                        f"Loaded {len(prior_reviews)} prior reviews."
                    )
                else:
                    st.success("✅ Similarity Search + LLM Analysis Complete!")

            except Exception as exc:
                logging.error("Pipeline error: %s", exc, exc_info=True)
                unified_progress.progress(0.0, text=f"❌ Pipeline failed: {str(exc)[:60]}")
                st.error(f"❌ An error occurred: {exc}")
                return

    # ── Display results ───────────────────────────────────────────────────
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        base_data = st.session_state.base_data
        user_data = st.session_state.user_data
        base_skipped, user_skipped = st.session_state.stats

        summary_col, vis_col = st.columns(2)
        with vis_col:
            st.subheader("📊 Embedding Visualization")
            fig = plot_embeddings(
                st.session_state.base_embeddings,
                st.session_state.user_embeddings,
                base_data,
                user_data,
            )
            st.plotly_chart(fig, use_container_width=True)
        with summary_col:
            st.subheader("📈 Summary")
            display_summary(df.to_dict("records"))

        # ── Filter panel ──────────────────────────────────────────────────
        filtered_df = render_filter_panel(df)

        # ── Results table ─────────────────────────────────────────────────
        st.subheader("📋 Results Table")
        note = (
            f"**Note**: Skipped {base_skipped} base and {user_skipped} "
            "query entries due to empty or invalid text."
        )
        if "Similarity_Level" in df.columns:
            tokens = st.session_state.total_tokens
            total_token_count = tokens["prompt_tokens"] + tokens["completion_tokens"]
            note += (
                f"<br>📊 **Total LLM tokens used**: {total_token_count} "
                f"({tokens['prompt_tokens']} prompt + {tokens['completion_tokens']} completion)"
            )
        note += f"<br>🔖 **Prompt version**: {st.session_state.get('active_prompt_version', 'v1')}"
        st.markdown(note, unsafe_allow_html=True)

        # Convert filtered_df to JSON for the iframe component
        df_json = filtered_df.to_json(orient="records")
        
        # Render the custom interactive Excel-like HTML/JS report viewer component
        state = _report_viewer(
            df_json=df_json,
            review_statuses=st.session_state.review_statuses,
            review_remarks=st.session_state.review_remarks,
            key="interactive_report_viewer",
            height=600
        )
        
        # Handle state updates and auto-save reviews to ChromaDB (disruption recovery)
        if state is not None:
            updated_statuses = state.get("review_statuses", {})
            updated_remarks = state.get("review_remarks", {})
            
            from am_ais_assist.pipeline import model as _pipeline_model
            results_by_qid = {
                str(r.get("Query_Object_Identifier", "")): r for r in df.to_dict("records")
            }
            user_id = st.session_state.get("user_id", st.session_state.get("user_session_id", ""))
            session_id = st.session_state.get("user_session_id", "")
            user_name = st.session_state.get("user_name", "unknown")
            base_hash = st.session_state.get("base_file_hash", "")
            check_hash = st.session_state.get("check_file_hash", "")
            prompt_ver = st.session_state.get("active_prompt_version", "v1")
            
            changed_detected = False
            for qid, status in updated_statuses.items():
                old_status = st.session_state.review_statuses.get(qid, "Pending")
                old_remark = st.session_state.review_remarks.get(qid, "")
                new_remark = updated_remarks.get(qid, "")
                
                if old_status != status or old_remark != new_remark:
                    st.session_state.review_statuses[qid] = status
                    st.session_state.review_remarks[qid] = new_remark
                    changed_detected = True
                    
                    if qid in results_by_qid:
                        orig = results_by_qid[qid]
                        ai_level = str(orig.get("Similarity_Level", ""))
                        if status != "Pending" and ai_level not in ("Exact Match", "Below Threshold") and _pipeline_model is not None:
                            try:
                                feedback_store.save_feedback(
                                    user_id=user_id,
                                    user_name=user_name,
                                    session_id=session_id,
                                    query_id=qid,
                                    matched_id=str(orig.get("Matched_Object_Identifier", "")),
                                    query_text=str(orig.get("Query_Sentence", ""))[:500],
                                    matched_text=str(orig.get("Matched_Sentence", ""))[:500],
                                    query_cleaned=str(orig.get("Query_Sentence_Cleaned_text", ""))[:300],
                                    matched_cleaned=str(orig.get("Matched_Sentence_Cleaned_text", ""))[:300],
                                    ai_score=orig.get("Similarity_Score", 0),
                                    ai_level=ai_level,
                                    ai_remark=str(orig.get("Remark", ""))[:300],
                                    verdict=status,
                                    base_file_hash=base_hash,
                                    check_file_hash=check_hash,
                                    prompt_version=prompt_ver,
                                    model=_pipeline_model,
                                    user_remark=new_remark,
                                )
                                logging.debug(f"Auto-saved verdict/remark for qid={qid} via component update")
                            except Exception as _save_exc:
                                logging.warning("Could not auto-save feedback: %s", _save_exc)
            
            if state.get("action") == "save":
                corrections_list = []
                for qid, verdict in st.session_state.review_statuses.items():
                    if verdict != "Pending":
                        orig = results_by_qid.get(qid)
                        if orig:
                            ai_level = str(orig.get("Similarity_Level", ""))
                            if ai_level not in ("Exact Match", "Below Threshold"):
                                corrections_list.append({
                                    "base_text": str(orig.get("Query_Sentence", "")),
                                    "new_text": str(orig.get("Matched_Sentence", "")),
                                    "ai_level": ai_level,
                                    "ai_score": orig.get("Similarity_Score", 0),
                                    "verdict": verdict,
                                    "user_remark": st.session_state.review_remarks.get(qid, "")
                                })
                if corrections_list:
                    import threading
                    from am_ais_assist.skill_generator import extract_and_update_skills
                    threading.Thread(
                        target=extract_and_update_skills,
                        args=(user_id, corrections_list),
                        daemon=True
                    ).start()
                    st.success("💾 Saved successfully!")
                else:
                    st.success("💾 Saved reviews successfully.")

            
            if changed_detected:
                st.rerun()

        # ── Download section ──────────────────────────────────────────────
        st.subheader("📥 Download Results")
        st.markdown("Download the results with LLM analysis in your preferred format.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="download-card">', unsafe_allow_html=True)
            st.download_button(
                label="📊 Download Excel Results",
                data=create_highlighted_excel(df, base_data, user_data),
                file_name="Similarity_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="download-card">', unsafe_allow_html=True)
            # Create Excel with reviews mapped
            df_with_review = df.copy()
            df_with_review["Review_Status"] = df_with_review["Query_Object_Identifier"].map(
                lambda qid: st.session_state.review_statuses.get(str(qid), "Pending")
            )
            df_with_review["User_Remark"] = df_with_review["Query_Object_Identifier"].map(
                lambda qid: st.session_state.review_remarks.get(str(qid), "")
            )
            excel_bytes = create_highlighted_excel(
                df_with_review,
                st.session_state.base_data or [],
                st.session_state.user_data or [],
                include_review_status=True,
            )
            st.download_button(
                label="📊 Download Excel with Reviews",
                data=excel_bytes,
                file_name="Similarity_Results_Reviewed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_with_review_download",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="download-card">', unsafe_allow_html=True)
            st.download_button(
                label="💾 Download JSON Results",
                data=df.to_json(orient="records", indent=2),
                file_name="similarity_results_with_llm.json",
                mime="application/json",
                key="json_download",
            )
            st.markdown("</div>", unsafe_allow_html=True)


        # ── Admin panel (visible only to admins) ──────────────────────────
        render_admin_panel()


if __name__ == "__main__":
    main()
