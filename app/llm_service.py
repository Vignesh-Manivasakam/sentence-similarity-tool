import os
import json
import logging
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib
import tiktoken
from app.config import (
    LLM_MODEL, LLM_CACHE_FILE, SYSTEM_PROMPT_PATH, LLM_BATCH_TOKEN_LIMIT
)

logger = logging.getLogger(__name__)

# Initialize Hugging Face pipeline for local LLM
try:
    client = pipeline("text-generation", model=LLM_MODEL, device=-1)  # CPU for HF Spaces free tier
    logger.info(f"Successfully initialized {LLM_MODEL} pipeline")
except Exception as e:
    logger.error(f"Failed to initialize {LLM_MODEL} pipeline: {e}")
    client = None

# Load system prompt from external file for maintainability
try:
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    logger.error(f"System prompt file not found at: {SYSTEM_PROMPT_PATH}")
    SYSTEM_PROMPT_TEMPLATE = "Error: System prompt could not be loaded."  # Fallback

def compute_prompt_hash(prompt: str) -> str:
    """Compute a hash for a prompt."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def get_user_llm_cache_file(user_session_id):
    """Get user-specific LLM cache file path"""
    from app.config import get_user_cache_dir
    user_cache_dir = get_user_cache_dir(user_session_id)
    return os.path.join(user_cache_dir, "llm_results_cache.json")

def load_llm_cache(user_session_id=None):
    """Load cached LLM results for specific user."""
    if user_session_id is None:
        # Fallback to global cache for backward compatibility
        cache_file = LLM_CACHE_FILE
    else:
        cache_file = get_user_llm_cache_file(user_session_id)
    
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load LLM cache '{cache_file}': {e}")
        return {}

def save_llm_cache(cache, user_session_id=None):
    """Save LLM results to user-specific cache."""
    if user_session_id is None:
        cache_file = LLM_CACHE_FILE
    else:
        cache_file = get_user_llm_cache_file(user_session_id)
    
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)
    except (TypeError, IOError) as e:
        logger.error(f"Failed to save LLM cache: {e}")

def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken for better accuracy."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback for when tiktoken might fail
        return len(text) // 4

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RuntimeError, ValueError))
)
def _call_llm_api(full_prompt: str, num_pairs: int) -> dict:
    """Helper function to make the LLM call using Hugging Face pipeline and handle responses."""
    content, prompt_tokens, completion_tokens = "", 0, 0
    try:
        prompt_tokens = estimate_tokens(full_prompt)
        response = client(full_prompt, max_new_tokens=512, return_full_text=False, temperature=0.0)
        content = response[0]['generated_text']
        completion_tokens = estimate_tokens(content)
        
        # Parse JSON output (assuming model outputs JSON as per prompt)
        json_str = content.strip().lstrip('```json').rstrip('```').strip()
        analysis = json.loads(json_str)

        if not isinstance(analysis, list) or len(analysis) != num_pairs:
            raise ValueError(f"LLM response length mismatch. Expected {num_pairs}, got {len(analysis)}")

        results = [
            {
                'Similarity_Score': result.get('score', 'Error'),
                'Similarity_Level': result.get('relationship', 'Parse Error'),
                'Remark': result.get('remark', 'No specific difference analysis provided.')
            }
            for result in analysis
        ]
        return {'results': results, 'tokens_used': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}

    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        error_type = type(e).__name__
        logger.error(f"LLM call failed due to {error_type}: {e}. Response: '{content}'")
        error_msg = f"{error_type} Error"
        error_result = {'Similarity_Score': 'Error', 'Similarity_Level': error_msg, 'Remark': 'Error during analysis'}
        return {'results': [error_result] * num_pairs, 'tokens_used': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}
    except Exception as e:
        logger.error(f"An unexpected LLM error occurred: {e}. Response: '{content}'")
        error_result = {'Similarity_Score': 'Error', 'Similarity_Level': 'LLM Call Failed', 'Remark': 'Model error occurred'}
        return {'results': [error_result] * num_pairs, 'tokens_used': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}

def _process_batch(batch_pairs: list, cache: dict, user_session_id=None) -> dict:
    """Processes a single batch of sentence pairs against the LLM, using caching."""
    batch_prompt_body = ""
    for i, (s1, s2) in enumerate(batch_pairs):
        batch_prompt_body += f"Pair {i+1}:\nSentence 1: \"{s1}\"\nSentence 2: \"{s2}\"\n"

    full_prompt = SYSTEM_PROMPT_TEMPLATE + "\n" + batch_prompt_body
    cache_key = compute_prompt_hash(full_prompt)

    if cache_key in cache:
        logger.info(f"Returning cached LLM result for batch key: {cache_key[:10]}...")
        return cache[cache_key]

    logger.info(f"Calling LLM for a batch of {len(batch_pairs)} pairs.")
    response = _call_llm_api(full_prompt, len(batch_pairs))
    
    # Only cache successful, valid results
    if all(res.get('Similarity_Level') not in ['Error', 'Parse Error'] for res in response['results']):
        cache[cache_key] = response
        
    return response

def get_llm_analysis_batch(sentence_pairs: list, user_session_id=None) -> dict:
    """
    Analyzes sentence pairs by iteratively creating batches that respect token limits.
    This avoids recursion errors and is more robust for production.
    """
    if not client:
        error_result = {'Similarity_Score': 'Error', 'Similarity_Level': 'LLM Model Not Configured', 'Remark': 'Model initialization failed'}
        return {'results': [error_result] * len(sentence_pairs), 'tokens_used': {}}

    cache = load_llm_cache(user_session_id)
    all_results = [None] * len(sentence_pairs)
    total_tokens_used = {'prompt_tokens': 0, 'completion_tokens': 0}

    base_prompt_tokens = estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
    current_batch_pairs = []
    current_batch_indices = []
    current_batch_tokens = base_prompt_tokens

    for i, pair in enumerate(sentence_pairs):
        pair_text = f"Pair {len(current_batch_pairs)+1}:\nSentence 1: \"{pair[0]}\"\nSentence 2: \"{pair[1]}\"\n"
        pair_tokens = estimate_tokens(pair_text)
        
        if current_batch_pairs and current_batch_tokens + pair_tokens > LLM_BATCH_TOKEN_LIMIT:
            batch_response = _process_batch(current_batch_pairs, cache, user_session_id)
            for j, res in enumerate(batch_response['results']):
                all_results[current_batch_indices[j]] = res
            
            total_tokens_used['prompt_tokens'] += batch_response['tokens_used'].get('prompt_tokens', 0)
            total_tokens_used['completion_tokens'] += batch_response['tokens_used'].get('completion_tokens', 0)
            
            current_batch_pairs, current_batch_indices, current_batch_tokens = [], [], base_prompt_tokens

        current_batch_pairs.append(pair)
        current_batch_indices.append(i)
        current_batch_tokens += pair_tokens

    if current_batch_pairs:
        batch_response = _process_batch(current_batch_pairs, cache, user_session_id)
        for j, res in enumerate(batch_response['results']):
            all_results[current_batch_indices[j]] = res
            
        total_tokens_used['prompt_tokens'] += batch_response['tokens_used'].get('prompt_tokens', 0)
        total_tokens_used['completion_tokens'] += batch_response['tokens_used'].get('completion_tokens', 0)
    
    save_llm_cache(cache, user_session_id)
    
    # Sanity check to ensure no results were missed
    if None in all_results:
        logger.error("LLM analysis resulted in missing data points. Filling with errors.")
        error_result = {'Similarity_Score': 'Error', 'Similarity_Level': 'Processing Error', 'Remark': 'Results missing'}
        all_results = [res if res is not None else error_result for res in all_results]

    return {'results': all_results, 'tokens_used': total_tokens_used}