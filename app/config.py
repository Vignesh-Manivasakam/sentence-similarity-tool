import os
import uuid
from datetime import datetime, timedelta

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- File and Cache Paths ----
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'cache')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')
PROMPT_DIR = os.path.join(BASE_DIR, 'app', 'prompts')

# Cache files
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_results_cache.json")
BASE_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "base_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "base_index.faiss")
HASH_FILE = os.path.join(CACHE_DIR, "base_file_hash.txt")

# ---- Embedding Model Configuration (HuggingFace) ----
# Using BGE-large-en-v1.5 - state-of-the-art open-source embedding model
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'  # Or 'BAAI/bge-large-en' for older version
EMBEDDING_DIMENSION = 1024  # BGE-large produces 1024-dim embeddings
EMBEDDING_BATCH_SIZE = 32  # Adjust based on your GPU/CPU
EMBEDDING_DEVICE = 'cuda'  # Change to 'cpu' if no GPU available

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
LOG_LEVEL = 'INFO'

# ---- LLM Configuration (Groq) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set via environment variable
GROQ_MODEL = "openai/gpt-oss-20b"  # As per your example
GROQ_TEMPERATURE = 0.0  # Lower for deterministic outputs
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
                dir_time = datetime.fromtimestamp(os.path.getctime(user_dir))
                if dir_time < cutoff_time:
                    import shutil
                    shutil.rmtree(user_dir)
                    print(f"Cleaned up expired session: {item}")

def get_active_user_count():
    """Get number of active user sessions"""
    if not os.path.exists(CACHE_DIR):
        return 0
    
    return len([d for d in os.listdir(CACHE_DIR) if d.startswith('user_')])