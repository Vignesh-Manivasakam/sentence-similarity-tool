import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import tempfile
import pandas as pd
import streamlit as st
from app import core  

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.handlers.remove(handler)

# Set the log level from config
from app.config import LOG_LEVEL

# ✅ FIX: Use temp directory for HuggingFace Spaces compatibility
log_dir = os.path.join(tempfile.gettempdir(), 'ais_logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

# Create handlers with error handling
try:
    file_handler = RotatingFileHandler(
        log_file, mode='a', maxBytes=5*1024*1024, backupCount=2, encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger('').addHandler(file_handler)
    logging.info(f"File logging enabled at: {log_file}")
except PermissionError as e:
    # If file logging fails (shouldn't happen in /tmp, but just in case)
    print(f"Warning: Could not create log file at {log_file}: {e}")
    print("Continuing with console logging only...")

# Console handler (always works)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger('').addHandler(console_handler)

# Configure root logger
logging.getLogger('').setLevel(LOG_LEVEL.upper())

logging.info("Logging configured successfully for HuggingFace Spaces deployment.")
logging.info(f"Log directory: {log_dir}")

# Now, set page config and import the rest of your app modules
st.set_page_config(layout="wide", page_title="AIS")

from app.config import (
    DEFAULT_THRESHOLDS, MAX_FILE_SIZE,
    LLM_ANALYSIS_MIN_THRESHOLD, LLM_PERFECT_MATCH_THRESHOLD
)
from app.pipeline import run_similarity_pipeline
from app.postprocess import display_summary, create_highlighted_excel, df_to_html_table
from app.utils import plot_embeddings
from app.llm_service import get_llm_analysis_batch


# Add these functions to app.py after the imports

def initialize_user_session():
    """Initialize user session if not exists"""
    if 'user_session_id' not in st.session_state:
        from app.config import generate_user_session
        st.session_state.user_session_id = generate_user_session()
        logging.info(f"New user session created: {st.session_state.user_session_id}")
    return st.session_state.user_session_id

def initialize_file_session():
    """Initialize file-related session state"""
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = False
    if 'base_file_data' not in st.session_state:
        st.session_state.base_file_data = None
    if 'check_file_data' not in st.session_state:
        st.session_state.check_file_data = None
    if 'base_columns' not in st.session_state:
        st.session_state.base_columns = []
    if 'check_columns' not in st.session_state:
        st.session_state.check_columns = []
    if 'last_base_file_name' not in st.session_state:
        st.session_state.last_base_file_name = None
    if 'last_check_file_name' not in st.session_state:
        st.session_state.last_check_file_name = None

def display_session_info():
    """Display current session information for debugging"""
    if st.sidebar.checkbox("🔍 Show Session Info (Debug)", value=False):
        user_session_id = st.session_state.get('user_session_id', 'Not initialized')
        st.sidebar.write(f"**Session ID:** `{user_session_id[:8]}...`")
        
        # Show cache directory info
        if user_session_id != 'Not initialized':
            from app.config import get_user_cache_dir, get_user_output_dir, get_active_user_count
            cache_dir = get_user_cache_dir(user_session_id)
            output_dir = get_user_output_dir(user_session_id)
            
            st.sidebar.write(f"**Cache Dir:** `{os.path.basename(cache_dir)}`")
            st.sidebar.write(f"**Output Dir:** `{os.path.basename(output_dir)}`")
            
            # Show cache files status
            from app.core import get_user_cache_files
            cache_files = get_user_cache_files(user_session_id)
            
            st.sidebar.write("**Cache Files:**")
            for name, path in cache_files.items():
                exists = "✅" if os.path.exists(path) else "❌"
                file_size = ""
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    file_size = f" ({size_mb:.1f}MB)" if size_mb > 0.1 else f" ({os.path.getsize(path)}B)"
                st.sidebar.write(f"  {exists} {name}{file_size}")
            
            # Show active users count
            active_users = get_active_user_count()
            st.sidebar.write(f"**Active Users:** {active_users}")

def cleanup_old_sessions():
    """Optional: Add a button to cleanup old sessions"""
    if st.sidebar.button("🧹 Cleanup Old Sessions"):
        from app.config import cleanup_expired_sessions
        try:
            cleanup_expired_sessions()
            st.sidebar.success("Cleanup completed!")
        except Exception as e:
            st.sidebar.error(f"Cleanup failed: {e}")

def load_sidebar():
    """Manages the Streamlit sidebar for file uploads, column selection, and settings."""
    with st.sidebar:
        st.header("📂 File Selection & Settings")
        with st.expander("Upload Files", expanded=True):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # --- Base File Logic with Session State ---
            base_file = st.file_uploader("Base Excel File 📑", type=["xlsx"])
            base_id_col, base_text_col, base_meta_cols = None, None, []
            
            if base_file:
                if base_file.size > MAX_FILE_SIZE:
                    st.error(f"Base file size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
                    base_file = None
                else:
                    # Check if we need to reprocess the base file
                    file_changed = st.session_state.last_base_file_name != base_file.name
                    
                    if file_changed or st.session_state.base_file_data is None:
                        try:
                            with st.spinner("Processing base file..."):
                                st.session_state.base_file_data = pd.read_excel(base_file)
                                st.session_state.base_columns = list(st.session_state.base_file_data.columns)
                                st.session_state.last_base_file_name = base_file.name
                                st.session_state.files_processed = True
                        except Exception as e:
                            st.error(f"Error reading base file: {e}")
                            base_file = None
                            st.session_state.base_file_data = None
                            st.session_state.base_columns = []
                    
                    # Use cached data for column selection
                    if st.session_state.base_file_data is not None:
                        st.markdown('<p class="upload-success">✅ Base file uploaded!</p>', unsafe_allow_html=True)
                        base_id_col = st.selectbox("Select Base Identifier Column", st.session_state.base_columns, key="base_id_col")
                        base_text_col = st.selectbox("Select Base Text Column", st.session_state.base_columns, key="base_text_col")
                        remaining_cols = [c for c in st.session_state.base_columns if c not in [base_id_col, base_text_col]]
                        base_meta_cols = st.multiselect("Select Base additional Columns (Optional)", remaining_cols, key="base_meta_cols")

            # --- Check File Logic with Session State ---
            check_file = st.file_uploader("Check Excel File 📝", type=["xlsx"])
            check_id_col, check_text_col, check_meta_cols = None, None, []
            
            if check_file:
                if check_file.size > MAX_FILE_SIZE:
                    st.error(f"Check file size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
                    check_file = None
                else:
                    # Check if we need to reprocess the check file
                    file_changed = st.session_state.last_check_file_name != check_file.name
                    
                    if file_changed or st.session_state.check_file_data is None:
                        try:
                            with st.spinner("Processing check file..."):
                                st.session_state.check_file_data = pd.read_excel(check_file)
                                st.session_state.check_columns = list(st.session_state.check_file_data.columns)
                                st.session_state.last_check_file_name = check_file.name
                                st.session_state.files_processed = True
                        except Exception as e:
                            st.error(f"Error reading check file: {e}")
                            check_file = None
                            st.session_state.check_file_data = None
                            st.session_state.check_columns = []
                    
                    # Use cached data for column selection
                    if st.session_state.check_file_data is not None:
                        st.markdown('<p class="upload-success">✅ Target file uploaded!</p>', unsafe_allow_html=True)
                        check_id_col = st.selectbox("Select Target Identifier Column", st.session_state.check_columns, key="check_id_col")
                        check_text_col = st.selectbox("Select Target Text Column", st.session_state.check_columns, key="check_text_col")
                        remaining_cols = [c for c in st.session_state.check_columns if c not in [check_id_col, check_text_col]]
                        check_meta_cols = st.multiselect("Select Target additional Columns (Optional)", remaining_cols, key="check_meta_cols")

            top_k = st.number_input("Top K Matches 🎯", min_value=1, max_value=10, value=3)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Relationship Classification Guide", expanded=False):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cols = st.columns(1)
            with cols[0]:
                st.markdown("""
                ✅ **Exact Match**: 1.0  
                *Perfect string*
                
                🟢 **Equivalent**: 0.95 - 1.0  
                *Same meaning, different wording*
                """)
                st.markdown("""
                🟡 **Related**: 0.50 - 0.94  
                *Related concepts, partial overlap*
                
                🔴 **Contradictory**: 0.00 - 0.20  
                *Opposite/conflicting meanings*
                """)
            st.markdown('</div>', unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Similarity Search")
        
    return base_file, check_file, top_k, run_btn, base_id_col, base_text_col, check_id_col, check_text_col, base_meta_cols, check_meta_cols
def main():
    """Main function to run the Streamlit application."""
    st.markdown("<h1 style='text-align: center;'>📘 AI Similarity Assist Tool🔍</h1>", unsafe_allow_html=True)
    # Initialize user session
    user_session_id = initialize_user_session()
    # ADD THIS LINE - Initialize file session state
    initialize_file_session()
    
    try:
        with open("static/css/custom.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

    # *** UPDATED: Removed thresholds from unpacking ***
    base_file, check_file, top_k, run_btn, base_id_col, base_text_col, check_id_col, check_text_col, base_meta_cols, check_meta_cols = load_sidebar()

    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}

    if run_btn:
        if not base_file or not check_file:
            st.warning("⚠️ Please upload both Excel files.")
            return
        if not base_id_col or not base_text_col or not check_id_col or not check_text_col:
            st.warning("⚠️ Please select identifier and text columns for both files.")
            return
    
        # ✅ Column validation
        if base_id_col == base_text_col:
            st.error("❌ **Base file error**: You selected the same column for both Identifier and Text. Please choose different columns!")
            return
    
        if check_id_col == check_text_col:
            st.error("❌ **Check file error**: You selected the same column for both Identifier and Text. Please choose different columns!")
            return
        
        # ========== UNIFIED PROGRESS PIPELINE ==========
        with st.spinner("Running integrated similarity search with LLM analysis..."):
            # *** NEW: Create unified progress bar ***
            unified_progress = st.progress(0, text="🚀 Initializing pipeline...")
            
            # *** NEW: Create progress manager ***
            from app.progress_manager import UnifiedProgressManager
            progress_manager = UnifiedProgressManager(unified_progress)
            
            try:
                # *** UPDATED: Pass progress manager instead of individual callbacks ***
                results, base_embeddings, user_embeddings, base_data, user_data, base_skipped_count, user_skipped_count, llm_tokens = run_similarity_pipeline(
                    base_file, check_file, top_k, None, progress_manager,  
                    base_id_col, base_text_col, check_id_col, check_text_col,
                    base_meta_cols, check_meta_cols, user_session_id
                )
                
                # Store results
                st.session_state.results_df = pd.DataFrame(results)
                st.session_state.base_data = base_data
                st.session_state.user_data = user_data
                st.session_state.base_embeddings = base_embeddings
                st.session_state.user_embeddings = user_embeddings
                st.session_state.stats = (base_skipped_count, user_skipped_count)
                st.session_state.total_tokens = llm_tokens
                
                # *** UPDATED: Final success message ***
                unified_progress.progress(1.0, text="🎉 All phases complete!")
                st.success("✅ Similarity Search + LLM Analysis Complete!")
                
            except Exception as e:
                st.error(f"❌ An error occurred during pipeline: {e}")
                logging.error(f"Pipeline error: {e}", exc_info=True)
                # *** NEW: Show error in progress bar ***
                if 'unified_progress' in locals():
                    unified_progress.progress(0.0, text=f"❌ Pipeline failed: {str(e)[:50]}...")
                return

    # ========== DISPLAY RESULTS ==========
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        base_data = st.session_state.base_data
        user_data = st.session_state.user_data
        base_skipped, user_skipped = st.session_state.stats

        summary_col, vis_col = st.columns(2)
        with vis_col:
            st.subheader("📊 Embedding Visualization")
            fig = plot_embeddings(st.session_state.base_embeddings, st.session_state.user_embeddings, base_data, user_data)
            st.plotly_chart(fig, use_container_width=True)
        with summary_col:
            st.subheader("📈 Summary")
            display_summary(df.to_dict('records'))

        # ========== RESULTS TABLE ==========
        st.subheader("📋 Results Table")
        note = f"**Note**: Skipped {base_skipped} base and {user_skipped} query entries due to empty or invalid text."
        if 'Similarity_Level' in df.columns:
            total_token_count = st.session_state.total_tokens['prompt_tokens'] + st.session_state.total_tokens['completion_tokens']
            note += f"<br>📊 **Total LLM tokens used**: {total_token_count} ({st.session_state.total_tokens['prompt_tokens']} prompt + {st.session_state.total_tokens['completion_tokens']} completion)"
        st.markdown(note, unsafe_allow_html=True)
        
        # Display table
        st.markdown(df_to_html_table(df, base_data, user_data), unsafe_allow_html=True)

        # ========== DOWNLOAD SECTION ==========
        st.subheader("📥 Download Results")
        st.markdown("Download the results table with LLM analysis in your preferred format.")
        download_container = st.container()
        with download_container:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="download-card">', unsafe_allow_html=True)
                st.download_button(
                    label="📊 Download Excel Results",
                    data=create_highlighted_excel(df, base_data, user_data),
                    file_name="Similarity_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="excel_download"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="download-card">', unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download JSON Results",
                    data=df.to_json(orient="records", indent=2),
                    file_name="similarity_results_llm.json",
                    mime="application/json",
                    key="json_download"
                )
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
