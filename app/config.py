import os
import uuid
import tempfile
from datetime import datetime, timedelta

# ✅ FIX: Use temporary directory for HuggingFace Spaces
# This ensures compatibility with read-only filesystems
TEMP_BASE_DIR = tempfile.gettempdir()
PROJECT_NAME = 'Similarity Assist Tool'

# Project root directory (for reference only, not for writing)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- File and Cache Paths (All in /tmp) ----
CACHE_DIR = os.path.join(TEMP_BASE_DIR, PROJECT_NAME, 'cache')
OUTPUT_DIR = os.path.join(TEMP_BASE_DIR, PROJECT_NAME, 'output')
PROMPT_DIR = os.path.join(BASE_DIR, 'app', 'prompts')  # Read-only, that's fine

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache files
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_results_cache.json")
BASE_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "base_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "base_index.faiss")
HASH_FILE = os.path.join(CACHE_DIR, "base_file_hash.txt")

# ---- Embedding Model Configuration (HuggingFace) ----
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = 'cuda' if os.getenv('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'

# Prompt file for maintainability
SYSTEM_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_prompt.txt')

# ---- Thresholds and Limits ----
DEFAULT_THRESHOLDS = {
    'exact': 1.0,
    'most': 0.9,
    'moderate': 0.7
}

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Centralized configuration
MAX_TOKENS_FOR_TRUNCATION = 512
LLM_BATCH_TOKEN_LIMIT = 100000

# ---- Logging ----
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ---- LLM Configuration (Groq) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TEMPERATURE = 0.0
GROQ_MAX_TOKENS = 8192
GROQ_REASONING_EFFORT = "medium"

# LLM Optimization
LLM_ANALYSIS_MIN_THRESHOLD = 0.4
LLM_PERFECT_MATCH_THRESHOLD = 0.999

# User session configuration
SESSION_TIMEOUT_HOURS = 24
MAX_CONCURRENT_USERS = 100

def generate_user_session():
    """Generate a unique session ID for each user"""
    return str(uuid.uuid4())

def get_user_cache_dir(user_session_id):
    """Get user-specific cache directory"""
    user_cache_dir = os.path.join(CACHE_DIR, f"user_{user_session_id}")
    os.makedirs(user_cache_dir, exist_ok=True)
    return user_cache_dir

def get_user_output_dir(user_session_id):
    """Get user-specific output directory"""
    user_output_dir = os.path.join(OUTPUT_DIR, f"user_{user_session_id}")
    os.makedirs(user_output_dir, exist_ok=True)
    return user_output_dir

def cleanup_expired_sessions():
    """Clean up expired user sessions"""
    if not os.path.exists(CACHE_DIR):
        return
    
    cutoff_time = datetime.now() - timedelta(hours=SESSION_TIMEOUT_HOURS)
    
    for item in os.listdir(CACHE_DIR):
        if item.startswith('user_'):
            user_dir = os.path.join(CACHE_DIR, item)
            if os.path.isdir(user_dir):
                try:
                    dir_time = datetime.fromtimestamp(os.path.getctime(user_dir))
                    if dir_time < cutoff_time:
                        import shutil
                        shutil.rmtree(user_dir)
                        print(f"Cleaned up expired session: {item}")
                except Exception as e:
                    print(f"Error cleaning up session {item}: {e}")

def get_active_user_count():
    """Get number of active user sessions"""
    if not os.path.exists(CACHE_DIR):
        return 0
    
    try:
        return len([d for d in os.listdir(CACHE_DIR) if d.startswith('user_') and os.path.isdir(os.path.join(CACHE_DIR, d))])
    except Exception:
        return 0

# ✅ Log the paths for debugging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Cache directory: {CACHE_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Embedding device: {EMBEDDING_DEVICE}")