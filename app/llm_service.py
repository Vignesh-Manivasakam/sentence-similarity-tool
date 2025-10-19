import os
import json
import logging
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import hashlib
import tiktoken
from app.config import (
    GROQ_API_KEY, GROQ_MODEL, LLM_CACHE_FILE,
    SYSTEM_PROMPT_PATH, LLM_BATCH_TOKEN_LIMIT,
    GROQ_TEMPERATURE, GROQ_MAX_TOKENS, GROQ_REASONING_EFFORT
)

logger = logging.getLogger(__name__)

# Initialize Groq client
client = None
try:
    if not GROQ_API_KEY or GROQ_API_KEY == "":
        logger.error("Groq API Key not configured. Please set GROQ_API_KEY in HuggingFace Space secrets.")
        logger.error("Go to: Settings → Repository secrets → Add GROQ_API_KEY")
    else:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")

# Load system prompt from external file
try:
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    logger.error(f"System prompt file not found at: {SYSTEM_PROMPT_PATH}")
    SYSTEM_PROMPT_TEMPLATE = """You are an expert text similarity analyzer. For each sentence pair provided, analyze their semantic similarity and provide:

1. A similarity score (0.0 to 1.0)
2. A relationship classification: "Exact Match", "Most Similar", "Moderately Similar", or "No Match"
3. A brief remark explaining the key differences or similarities

Format your response as a JSON array with one object per pair:

[
  {
    "score": 0.95,
    "relationship": "Most Similar",
    "remark": "Same core concept, minor wording differences"
  }
]

Analyze these sentence pairs:
"""

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
        return len(text) // 4

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,))
)
def _call_llm_api(full_prompt: str, num_pairs: int) -> dict:
    """Helper function to make the actual Groq API call and handle streaming responses."""
    content, prompt_tokens, completion_tokens = "", 0, 0
    
    # Check if client is initialized
    if client is None:
        logger.error("Groq client is not initialized. Check your API key configuration.")
        error_result = {
            'Similarity_Score': 'Error',
            'Similarity_Level': 'API Key Missing',
            'Remark': 'Groq API key not configured'
        }
        return {
            'results': [error_result] * num_pairs,
            'tokens_used': {'prompt_tokens': 0, 'completion_tokens': 0}
        }
    
    try:
        prompt_tokens = estimate_tokens(full_prompt)
        
        # ✅ FIX: Updated Groq API call with proper response handling
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            top_p=1,
            stream=True,
            stop=None
        )
        
        # ✅ FIX: Proper handling of streaming response
        for chunk in stream:
            # Handle different response structures
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                # Check if delta has content attribute
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                # Alternative: check if it's a dict
                elif isinstance(delta, dict) and 'content' in delta and delta['content']:
                    content += delta['content']
        
        completion_tokens = estimate_tokens(content)
        
        # Parse JSON response
        json_str = content.strip()
        # Remove markdown code blocks if present
        if json_str.startswith('```'):
            json_str = json_str.split('```')[1]
            if json_str.startswith('json'):
                json_str = json_str[4:]
            json_str = json_str.strip()
        
        analysis = json.loads(json_str)

        if not isinstance(analysis, list) or len(analysis) != num_pairs:
            logger.warning(f"LLM response length mismatch. Expected {num_pairs}, got {len(analysis) if isinstance(analysis, list) else 'non-list'}")
            # Pad or truncate as needed
            if isinstance(analysis, list):
                if len(analysis) < num_pairs:
                    # Pad with default values
                    default_result = {
                        'score': 0.5,
                        'relationship': 'Unknown',
                        'remark': 'Analysis incomplete'
                    }
                    analysis.extend([default_result] * (num_pairs - len(analysis)))
                else:
                    # Truncate
                    analysis = analysis[:num_pairs]

        results = [
            {
                'Similarity_Score': result.get('score', 'Error'),
                'Similarity_Level': result.get('relationship', 'Parse Error'),
                'Remark': result.get('remark', 'No specific difference analysis provided.')
            }
            for result in analysis
        ]
        return {
            'results': results,
            'tokens_used': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        }

    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON decode failed: {e}. Response: '{content[:500]}'")
        error_result = {
            'Similarity_Score': 'Error',
            'Similarity_Level': 'JSON Parse Error',
            'Remark': f'Could not parse LLM response'
        }
        return {
            'results': [error_result] * num_pairs,
            'tokens_used': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API request failed: {e}")
        error_result = {
            'Similarity_Score': 'Error',
            'Similarity_Level': 'API Request Failed',
            'Remark': 'Network or API error'
        }
        return {
            'results': [error_result] * num_pairs,
            'tokens_used': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        }
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}", exc_info=True)
        error_result = {
            'Similarity_Score': 'Error',
            'Similarity_Level': 'LLM Error',
            'Remark': f'Unexpected error: {type(e).__name__}'
        }
        return {
            'results': [error_result] * num_pairs,
            'tokens_used': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        }

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

    logger.info(f"Calling Groq LLM for a batch of {len(batch_pairs)} pairs.")
    response = _call_llm_api(full_prompt, len(batch_pairs))
    
    # Only cache successful, valid results
    if all(res.get('Similarity_Level') not in ['Error', 'Parse Error', 'API Key Missing'] for res in response['results']):
        cache[cache_key] = response
        
    return response

def get_llm_analysis_batch(sentence_pairs: list, user_session_id=None) -> dict:
    """
    Analyzes sentence pairs by iteratively creating batches that respect token limits.
    """
    if not client:
        logger.warning("Groq client not initialized. LLM analysis will be skipped.")
        error_result = {
            'Similarity_Score': 'Error',
            'Similarity_Level': 'LLM API Key Not Configured',
            'Remark': 'Please add GROQ_API_KEY to HuggingFace Space secrets'
        }
        return {
            'results': [error_result] * len(sentence_pairs),
            'tokens_used': {'prompt_tokens': 0, 'completion_tokens': 0}
        }

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
    
    # Sanity check
    if None in all_results:
        logger.error("LLM analysis resulted in missing data points. Filling with errors.")
        error_result = {
            'Similarity_Score': 'Error',
            'Similarity_Level': 'Processing Error',
            'Remark': 'Results missing'
        }
        all_results = [res if res is not None else error_result for res in all_results]

    return {'results': all_results, 'tokens_used': total_tokens_used}