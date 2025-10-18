import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import logging
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.postprocess import highlight_word_differences
from app.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE, BASE_EMBEDDINGS_FILE, FAISS_INDEX_FILE, HASH_FILE,
    MAX_TOKENS_FOR_TRUNCATION
)

logger = logging.getLogger(__name__)

# Global embedding progress callback
_EMBEDDING_PROGRESS_CB = None

def set_embedding_progress_callback(cb):
    """
    Register a callback that accepts (completed_batches: int, total_batches: int).
    Pass None to disable.
    """
    global _EMBEDDING_PROGRESS_CB
    _EMBEDDING_PROGRESS_CB = cb

def get_user_cache_files(user_session_id):
    """Get user-specific cache file paths"""
    from app.config import get_user_cache_dir
    user_cache_dir = get_user_cache_dir(user_session_id)
    return {
        'embeddings': os.path.join(user_cache_dir, "base_embeddings.npy"),
        'index': os.path.join(user_cache_dir, "base_index.faiss"),
        'hash': os.path.join(user_cache_dir, "base_file_hash.txt")
    }

class EmbeddingModel:
    """
    Wrapper class for HuggingFace Sentence-Transformers (BGE-large).
    """

    def __init__(self, model_name, batch_size=32, device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded HuggingFace model: {model_name} on {device}")
        logger.info(f"Embedding dimension: {self._embedding_dimension}")

    @property
    def tokenizer(self):
        """
        Expose the underlying tokenizer from SentenceTransformer.
        The tokenizer is accessed from the internal Transformer module.
        """
        try:
            # Method 1: Direct tokenizer attribute
            if hasattr(self.model, 'tokenizer'):
                return self.model.tokenizer
            
            # Method 2: Access from the first module (Transformer model)
            if len(self.model) > 0:
                first_module = self.model[0]
                if hasattr(first_module, 'tokenizer'):
                    return first_module.tokenizer
                # Try auto_model attribute
                if hasattr(first_module, 'auto_model') and hasattr(first_module.auto_model, 'tokenizer'):
                    return first_module.auto_model.tokenizer
            
            logger.warning("Could not find tokenizer in SentenceTransformer model")
            return None
            
        except Exception as e:
            logger.error(f"Error accessing tokenizer: {e}")
            return None

    def truncate_text(self, text, max_tokens=None):
        """
        Truncate text to max tokens using the model's tokenizer if available.
        Falls back to character-based truncation if tokenizer is not available.
        
        Args:
            text (str): Text to truncate
            max_tokens (int): Maximum number of tokens (default: MAX_TOKENS_FOR_TRUNCATION from config)
        
        Returns:
            tuple: (truncated_text, was_truncated)
        """
        if not text:
            return text, False
        
        if max_tokens is None:
            max_tokens = MAX_TOKENS_FOR_TRUNCATION
        
        # Try tokenizer-based truncation
        tokenizer = self.tokenizer
        if tokenizer is not None:
            try:
                # Encode without special tokens
                tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                
                if len(tokens) > max_tokens:
                    # Truncate tokens
                    truncated_tokens = tokens[:max_tokens]
                    # Decode back to text
                    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    logger.debug(f"Tokenizer truncation: {len(tokens)} tokens → {max_tokens} tokens")
                    return truncated_text, True
                
                return text, False
                
            except Exception as e:
                logger.warning(f"Tokenizer truncation failed: {e}. Using character-based fallback.")
        
        # Fallback: character-based truncation (estimate ~4 chars/token for English)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            truncated = text[:max_chars]
            logger.debug(f"Character-based truncation: {len(text)} chars → {max_chars} chars")
            return truncated, True
        
        return text, False

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=None):
        """
        Encode texts into embeddings with deduplication and progress tracking.
        
        Args:
            texts: Single string or list of strings to encode
            convert_to_numpy: Whether to return numpy array
            show_progress_bar: Whether to show progress (handled via callback)
            batch_size: Batch size for encoding (default: self.batch_size)
        
        Returns:
            numpy.ndarray or list of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([]) if convert_to_numpy else []

        # Deduplicate inputs to avoid redundant encoding
        unique_texts = list(dict.fromkeys(texts))
        batch_sz = batch_size or self.batch_size
        
        logger.info(f"Encoding {len(unique_texts):,} unique texts (batch_size={batch_sz})")
        
        # Calculate total batches for progress tracking
        total_batches = (len(unique_texts) + batch_sz - 1) // batch_sz
        
        if _EMBEDDING_PROGRESS_CB:
            try:
                _EMBEDDING_PROGRESS_CB(0, total_batches)
            except Exception:
                pass

        # Encode with progress tracking
        embeddings = []
        for i in range(0, len(unique_texts), batch_sz):
            batch = unique_texts[i:i+batch_sz]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # Update progress
            completed = (i // batch_sz) + 1
            if _EMBEDDING_PROGRESS_CB:
                try:
                    _EMBEDDING_PROGRESS_CB(completed, total_batches)
                except Exception:
                    pass
            logger.info(f"Completed {completed}/{total_batches} batches")

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        
        # Create mapping from unique texts to embeddings
        text_to_embedding = {t: emb for t, emb in zip(unique_texts, all_embeddings)}
        
        # Map back to original order (including duplicates)
        final_embeddings = [text_to_embedding[t] for t in texts]
        
        return np.array(final_embeddings) if convert_to_numpy else final_embeddings

    def get_sentence_embedding_dimension(self):
        """Get the embedding dimension of the model."""
        return self._embedding_dimension


def load_model():
    """Load the HuggingFace BGE-large embedding model."""
    try:
        model = EmbeddingModel(
            model_name=EMBEDDING_MODEL_NAME,
            batch_size=EMBEDDING_BATCH_SIZE,
            device=EMBEDDING_DEVICE
        )
        logger.info("Successfully loaded HuggingFace embedding model.")
        return model
    except Exception as e:
        logger.error(f"FATAL: Failed to load embedding model. Error: {e}")
        raise RuntimeError(f"Could not load embedding model. Details: {e}")


def _compute_file_hash(file_obj):
    """Compute SHA256 hash of file for cache validation."""
    sha256 = hashlib.sha256()
    file_obj.seek(0)
    for chunk in iter(lambda: file_obj.read(4096), b""):
        sha256.update(chunk)
    file_obj.seek(0)
    return sha256.hexdigest()


def _check_cache_validity(current_hash: str, user_session_id=None) -> bool:
    """Check cache validity for specific user or global cache"""
    if user_session_id is None:
        cache_files = {
            'embeddings': BASE_EMBEDDINGS_FILE,
            'index': FAISS_INDEX_FILE,
            'hash': HASH_FILE
        }
    else:
        cache_files = get_user_cache_files(user_session_id)
    
    if not all(os.path.exists(f) for f in cache_files.values()):
        logger.info(f"Cache files missing for {user_session_id or 'global'}")
        return False
    
    try:
        with open(cache_files['hash'], "r", encoding='utf-8') as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            logger.info(f"Cache is valid for {user_session_id or 'global'}")
            return True
        logger.info(f"Cache hash mismatch for {user_session_id or 'global'}. Regenerating index.")
        return False
    except IOError as e:
        logger.warning(f"Error reading cache hash file for {user_session_id or 'global'}: {e}")
        return False


def _load_from_cache(user_session_id=None):
    """Load from user-specific cache or global cache"""
    try:
        if user_session_id is None:
            cache_files = {
                'embeddings': BASE_EMBEDDINGS_FILE,
                'index': FAISS_INDEX_FILE
            }
        else:
            cache_files = get_user_cache_files(user_session_id)
        
        logger.info(f"Loading FAISS index and embeddings from cache for {user_session_id or 'global'}")
        embeddings = np.load(cache_files['embeddings'])
        index = faiss.read_index(cache_files['index'])
        return index, embeddings
    except Exception as e:
        logger.warning(f"Failed to load from cache for {user_session_id or 'global'}: {e}. Regeneration will occur.")
        return None, None


def _generate_and_save_index(data: list, model, current_hash: str, user_session_id=None, progress_callback=None):
    """Generates new embeddings and FAISS index using BGE-large model, then saves them to user-specific or global cache."""
    logger.info(f"Generating new embeddings and FAISS index using BGE-large for {user_session_id or 'global'}")
    texts = [entry['Cleaned_Text'] for entry in data]
    if not texts:
        raise ValueError("No texts found for embedding generation")

    # Progress callback for embeddings
    if progress_callback:
        progress_callback(0.2, 1.0, f"🔗 Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Log embedding information
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"Number of texts embedded: {embeddings.shape[0]}")
    logger.info(f"Memory usage (approx): {embeddings.nbytes / 1024 / 1024:.2f} MB")

    if progress_callback:
        progress_callback(0.9, 1.0, "🏗️ Building FAISS index...")

    # Normalize for cosine similarity
    normalized = normalize(embeddings, axis=1, norm='l2')
    dim = normalized.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    logger.info(f"FAISS index created with dimension: {dim}")

    if progress_callback:
        progress_callback(0.95, 1.0, "💾 Saving to cache...")

    # Cache with user-specific or global paths
    try:
        if user_session_id is None:
            cache_files = {
                'embeddings': BASE_EMBEDDINGS_FILE,
                'index': FAISS_INDEX_FILE,
                'hash': HASH_FILE
            }
        else:
            cache_files = get_user_cache_files(user_session_id)
        
        os.makedirs(os.path.dirname(cache_files['embeddings']), exist_ok=True)
        
        np.save(cache_files['embeddings'], normalized)
        faiss.write_index(index, cache_files['index'])
        with open(cache_files['hash'], "w", encoding='utf-8') as f:
            f.write(current_hash)
        logger.info(f"Successfully cached embeddings and index for {user_session_id or 'global'}")
    except (IOError, Exception) as e:
        logger.warning(f"Failed to cache embeddings/index for {user_session_id or 'global'}: {e}")

    if progress_callback:
        progress_callback(1.0, 1.0, "✅ Index creation complete!")

    return index, normalized


def create_faiss_index(data: list, model, base_file_obj, user_session_id=None, progress_callback=None):
    """Create or load FAISS index for similarity search with user-specific caching."""
    if user_session_id:
        cache_files = get_user_cache_files(user_session_id)
        os.makedirs(os.path.dirname(cache_files['index']), exist_ok=True)
        logger.info(f"Using user-specific cache for session: {user_session_id}")
    else:
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        logger.info("Using global cache")
    
    if progress_callback:
        progress_callback(0, 3, "🔍 Computing file hash...")
    
    current_hash = _compute_file_hash(base_file_obj)

    if progress_callback:
        progress_callback(1, 3, "📋 Checking cache validity...")

    if _check_cache_validity(current_hash, user_session_id):
        index, embeddings = _load_from_cache(user_session_id)
        if index is not None and embeddings is not None:
            logger.info(f"Using cached FAISS index and embeddings for {user_session_id or 'global'}")
            if progress_callback:
                progress_callback(3, 3, "📦 Loaded from cache")
            return index, embeddings, data

    if progress_callback:
        progress_callback(2, 3, "⚡ Generating new index...")

    logger.info(f"Generating new FAISS index with BGE-large embeddings for {user_session_id or 'global'}")
    
    def index_generation_progress(sub_progress, sub_total, sub_message):
        if sub_total > 0:
            overall_progress = 2.0 + (sub_progress / sub_total)
            progress_callback(overall_progress, 3, sub_message)
    
    index, embeddings = _generate_and_save_index(data, model, current_hash, user_session_id, index_generation_progress)
    
    if progress_callback:
        progress_callback(3, 3, "🎉 Index generation complete")
    
    return index, embeddings, data


# ---------------------------
# Hierarchy tie-breaker functions
# ---------------------------

def _parse_hierarchy(hstr):
    """Parse hierarchy string into list of integers."""
    if not hstr:
        return []
    parts = hstr.replace('-', '.').split('.')
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
    return out


def _hierarchy_distance(qh, bh):
    """Calculate hierarchical distance between two hierarchy strings."""
    q = _parse_hierarchy(qh)
    b = _parse_hierarchy(bh)
    if not q or not b:
        return (0, float('inf'))
    lcp = 0
    for x, y in zip(q, b):
        if x == y:
            lcp += 1
        else:
            break
    diff = abs(q[lcp] - b[lcp]) if lcp < min(len(q), len(b)) else abs(len(q) - len(b))
    return (-lcp, diff)


def choose_by_hierarchy(query_entry, candidates):
    """Choose the best matching candidate based on hierarchy proximity."""
    qh = query_entry.get("Hierarchy")
    if not qh or not candidates:
        return candidates[0]
    logger.info(f"[TieBreaker] Query hierarchy: {qh}, candidates: {[c.get('Hierarchy') for c in candidates]}")
    best, best_key = None, (float('inf'), float('inf'))
    for c in candidates:
        key = _hierarchy_distance(qh, c.get("Hierarchy"))
        logger.info(f"[TieBreaker] Comparing with {c.get('Hierarchy')} → distance {key}")
        if key < best_key:
            best, best_key = c, key
    logger.info(f"[TieBreaker] Chosen candidate: {best.get('Hierarchy')} for query {qh}")
    return best if best else candidates[0]


# ---------------------------
# Main similarity search
# ---------------------------

def search_similar(user_data, index, base_data, top_k, thresholds, model, progress_callback=None):
    """
    Perform similarity search using FAISS index with exact match optimization.
    
    Args:
        user_data: List of user query entries
        index: FAISS index
        base_data: List of base entries
        top_k: Number of top matches to return
        thresholds: Dictionary of similarity thresholds
        model: EmbeddingModel instance
        progress_callback: Optional callback for progress updates
    
    Returns:
        Tuple of (results, user_embeddings)
    """
    logger.info(f"Starting similarity search for {len(user_data)} queries against {len(base_data)} base items")
    results = []

    total_phases = 3
    current_phase = 0
    
    if progress_callback:
        progress_callback(current_phase, total_phases, "Initializing similarity search...")

    # Map Cleaned_Text → list of base candidates
    text_to_candidates = {}
    for b in base_data:
        text_to_candidates.setdefault(b['Cleaned_Text'], []).append(b)

    exact_matches = {}
    remaining_user_data, remaining_indices = [], []

    # ========== PHASE 1: EXACT MATCH ==========
    current_phase += 1
    if progress_callback:
        progress_callback(current_phase, total_phases, f"Processing exact matches ({len(user_data)} queries)...")

    for i, user_entry in enumerate(user_data):
        user_text = user_entry['Cleaned_Text']
        candidates = text_to_candidates.get(user_text, [])
        if candidates:
            chosen = candidates[0] if len(candidates) == 1 else choose_by_hierarchy(user_entry, candidates)
            query_words, match_words = highlight_word_differences(
                user_entry['Original_Text'], chosen['Original_Text']
            )
            result_item = {
                'Query_Object_Identifier': user_entry['Object_Identifier'],
                'Query_Sentence': user_entry['Original_Text'],
                'Query_Sentence_Cleaned_text': user_entry['Cleaned_Text'],
                'Query_Sentence_Highlighted': query_words,
                'Matched_Object_Identifier': chosen['Object_Identifier'],
                'Matched_Sentence': chosen['Original_Text'],
                'Matched_Sentence_Cleaned_text': chosen['Cleaned_Text'],
                'Matched_Sentence_Highlighted': match_words,
                'Similarity_Score': 1.0000,
                'Similarity_Level': "Exact Match"
            }
            # Add metadata
            for key, value in user_entry.items():
                if key not in ['Object_Identifier','Original_Text','Cleaned_Text','Truncated','Hierarchy']:
                    result_item[f'Query_{key}'] = value
            for key, value in chosen.items():
                if key not in ['Object_Identifier','Original_Text','Cleaned_Text','Truncated','Hierarchy']:
                    result_item[f'Matched_{key}'] = value
            exact_matches[i] = [result_item]
        else:
            remaining_user_data.append(user_entry)
            remaining_indices.append(i)

    exact_count = len([i for i in exact_matches if exact_matches[i] and exact_matches[i][0]['Similarity_Score'] == 1.0])
    logger.info(f"Phase 1 complete: {exact_count} exact matches found, {len(remaining_user_data)} queries need embedding search")

    # ========== PHASE 2: EMBEDDING SEARCH ==========
    current_phase += 1
    if remaining_user_data:
        if progress_callback:
            progress_callback(current_phase, total_phases, f"Processing embeddings ({len(remaining_user_data)} queries)...")
        
        logger.info(f"Processing {len(remaining_user_data)} queries with BGE-large embeddings")
        remaining_texts = [e['Cleaned_Text'] for e in remaining_user_data]
        user_embeddings = model.encode(remaining_texts, convert_to_numpy=True)
        user_embeddings = normalize(user_embeddings, axis=1, norm='l2')

        D, I = index.search(user_embeddings, top_k)
        
        for idx, rem_idx in enumerate(remaining_indices):
            user_entry = remaining_user_data[idx]
            exact_matches[rem_idx] = []
            scores, indices = D[idx], I[idx]

            candidate_items = [(float(scores[rank]), base_data[indices[rank]])
                               for rank in range(top_k) if indices[rank] < len(base_data)]

            grouped_by_text = {}
            for sc, m in candidate_items:
                grouped_by_text.setdefault(m['Cleaned_Text'], []).append((sc, m))

            for cleaned_text, returned_list in grouped_by_text.items():
                rep_score = max(sc for sc, _ in returned_list) if returned_list else 0.0
                matches_to_consider = text_to_candidates.get(cleaned_text, [m for _, m in returned_list])
                chosen = choose_by_hierarchy(user_entry, matches_to_consider) if len(matches_to_consider) > 1 else matches_to_consider[0]

                # Classify similarity level
                if abs(rep_score - 1.0) < 1e-6:
                    label = "Exact Match"
                elif rep_score >= thresholds['most']:
                    label = "Most Similar"
                elif rep_score >= thresholds['moderate']:
                    label = "Moderately Similar"
                else:
                    label = "No Match"

                query_words, match_words = highlight_word_differences(
                    user_entry['Original_Text'], chosen['Original_Text']
                )
                result_item = {
                    'Query_Object_Identifier': user_entry['Object_Identifier'],
                    'Query_Sentence': user_entry['Original_Text'],
                    'Query_Sentence_Cleaned_text': user_entry['Cleaned_Text'],
                    'Query_Sentence_Highlighted': query_words,
                    'Matched_Object_Identifier': chosen['Object_Identifier'],
                    'Matched_Sentence': chosen['Original_Text'],
                    'Matched_Sentence_Cleaned_text': chosen['Cleaned_Text'],
                    'Matched_Sentence_Highlighted': match_words,
                    'Similarity_Score': round(rep_score, 2),
                    'Similarity_Level': label
                }
                for key, value in user_entry.items():
                    if key not in ['Object_Identifier','Original_Text','Cleaned_Text','Truncated','Hierarchy']:
                        result_item[f'Query_{key}'] = value
                for key, value in chosen.items():
                    if key not in ['Object_Identifier','Original_Text','Cleaned_Text','Truncated','Hierarchy']:
                        result_item[f'Matched_{key}'] = value
                exact_matches[rem_idx].append(result_item)
        
        logger.info(f"Phase 2 complete: Embedding search processed {len(remaining_user_data)} queries")
    else:
        logger.info("Phase 2 skipped: No queries require embedding search")
        if progress_callback:
            progress_callback(current_phase, total_phases, "Embedding search skipped (all exact matches)")

    # ========== PHASE 3: COLLECT RESULTS ==========
    current_phase += 1
    if progress_callback:
        progress_callback(current_phase, total_phases, "Collecting and finalizing results...")

    for i in range(len(user_data)):
        if i in exact_matches:
            results.extend(exact_matches[i])

    full_user_embeddings = np.zeros((len(user_data), model.get_sentence_embedding_dimension()))
    if remaining_user_data:
        remaining_embeddings = model.encode([entry['Cleaned_Text'] for entry in remaining_user_data], 
                                            convert_to_numpy=True, show_progress_bar=False)
        for idx, remaining_idx in enumerate(remaining_indices):
            full_user_embeddings[remaining_idx] = remaining_embeddings[idx]

    embedding_count = len(remaining_user_data)
    logger.info(f"Search complete: {exact_count} exact matches, {embedding_count} via embeddings")
    logger.info(f"Optimization: Saved ~{exact_count} embedding calls by using string match")
    logger.info(f"Total results generated: {len(results)}")

    if progress_callback:
        progress_callback(total_phases, total_phases, f"Search complete! {len(results)} results generated")

    return results, full_user_embeddings