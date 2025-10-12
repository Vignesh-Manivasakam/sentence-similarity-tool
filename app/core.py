import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.postprocess import highlight_word_differences
from app.config import (
    HF_MODEL_NAME, BASE_EMBEDDINGS_FILE, FAISS_INDEX_FILE, HASH_FILE
)

logger = logging.getLogger(__name__)

# Add global, optional embedding progress callback
_EMBEDDING_PROGRESS_CB = None

def set_embedding_progress_callback(cb):
    """
    Register a callback that accepts (completed_batches: int, total_batches: int).
    Pass None to disable.
    """
    global _EMBEDDING_PROGRESS_CB
    _EMBEDDING_PROGRESS_CB = cb

class EmbeddingModel:
    """
    Wrapper class for SentenceTransformer embeddings with batching, deduplication, and parallelism.
    """
    def __init__(self, model_name, batch_size=256, max_workers=4):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self._embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _embed_batch(self, batch):
        try:
            embeddings = self.model.encode(batch, batch_size=len(batch), normalize_embeddings=True)
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [np.zeros(self._embedding_dimension) for _ in batch]

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=None):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([]) if convert_to_numpy else []

        # Deduplicate inputs
        unique_texts = list(dict.fromkeys(texts))
        text_to_embedding = {}
        batches = [unique_texts[i:i+(batch_size or self.batch_size)]
                   for i in range(0, len(unique_texts), (batch_size or self.batch_size))]
        
        logger.info(f"Encoding {len(unique_texts):,} unique texts in {len(batches)} batches "
                    f"(batch_size={batch_size or self.batch_size})")
        logger.info(f"Launching {len(batches)} batch jobs with up to {self.max_workers} parallel workers")

        total_batches = len(batches)
        if _EMBEDDING_PROGRESS_CB:
            try:
                _EMBEDDING_PROGRESS_CB(0, total_batches)
            except Exception:
                pass

        # Parallel batch requests
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._embed_batch, b): b for b in batches}
            for fut in as_completed(futures):
                batch = futures[fut]
                try:
                    embeddings = fut.result()
                    for t, emb in zip(batch, embeddings):
                        text_to_embedding[t] = emb
                except Exception as e:
                    logger.error(f"Batch embedding failed for {batch[:2]}... : {e}")
                    for t in batch:
                        text_to_embedding[t] = np.zeros(self._embedding_dimension)

                completed += 1
                if _EMBEDDING_PROGRESS_CB:
                    try:
                        _EMBEDDING_PROGRESS_CB(completed, total_batches)
                    except Exception:
                        pass
                logger.info(f"Completed {completed}/{total_batches} batches")

        # Map back to original order
        all_embeddings = [text_to_embedding.get(t, np.zeros(self._embedding_dimension)) for t in texts]
        return np.array(all_embeddings) if convert_to_numpy else all_embeddings

    def get_sentence_embedding_dimension(self):
        return self._embedding_dimension

def get_user_cache_files(user_session_id):
    """Get user-specific cache file paths"""
    from app.config import get_user_cache_dir
    user_cache_dir = get_user_cache_dir(user_session_id)
    return {
        'embeddings': os.path.join(user_cache_dir, "base_embeddings.npy"),
        'index': os.path.join(user_cache_dir, "base_index.faiss"),
        'hash': os.path.join(user_cache_dir, "base_file_hash.txt")
    }

def compute_file_hash(file_obj):
    """Compute SHA256 hash of file content"""
    file_obj.seek(0)
    sha256 = hashlib.sha256()
    while chunk := file_obj.read(8192):
        sha256.update(chunk)
    file_obj.seek(0)
    return sha256.hexdigest()

def load_model():
    """Load the embedding model"""
    try:
        model = EmbeddingModel(HF_MODEL_NAME)
        logger.info(f"Successfully loaded model: {HF_MODEL_NAME}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {HF_MODEL_NAME}: {e}")
        raise

def create_faiss_index(data, model, base_file, user_session_id=None, progress_callback=None):
    """Create or load FAISS index for base data"""
    cache_files = get_user_cache_files(user_session_id) if user_session_id else {
        'embeddings': BASE_EMBEDDINGS_FILE,
        'index': FAISS_INDEX_FILE,
        'hash': HASH_FILE
    }

    file_hash = compute_file_hash(base_file) if base_file else None
    cached_hash = None
    if os.path.exists(cache_files['hash']):
        with open(cache_files['hash'], 'r') as f:
            cached_hash = f.read().strip()

    if (os.path.exists(cache_files['embeddings']) and 
        os.path.exists(cache_files['index']) and 
        file_hash == cached_hash):
        logger.info("Loading cached FAISS index and embeddings")
        embeddings = np.load(cache_files['embeddings'])
        index = faiss.read_index(cache_files['index'])
        return index, embeddings, data

    logger.info("Creating new FAISS index")
    texts = [entry['Cleaned_Text'] for entry in data]
    if progress_callback:
        progress_callback(0, 2, "🏗️ Generating embeddings...")
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if progress_callback:
        progress_callback(1, 2, "🏗️ Building FAISS index...")

    os.makedirs(os.path.dirname(cache_files['embeddings']), exist_ok=True)
    np.save(cache_files['embeddings'], embeddings)
    faiss.write_index(index, cache_files['index'])
    if file_hash:
        with open(cache_files['hash'], 'w') as f:
            f.write(file_hash)

    if progress_callback:
        progress_callback(2, 2, "🏗️ FAISS index created!")
    
    return index, embeddings, data

def choose_by_hierarchy(user_entry, candidates):
    """Choose the best candidate based on hierarchy if available"""
    user_hierarchy = user_entry.get('Hierarchy')
    if not user_hierarchy:
        return candidates[0]
    
    for candidate in candidates:
        if candidate.get('Hierarchy') == user_hierarchy:
            return candidate
    return candidates[0]

def search_similar(user_data, index, base_data, top_k, thresholds, model, progress_callback=None):
    """Search for similar sentences using FAISS and exact matching"""
    results = []
    exact_matches = {}
    remaining_user_data = []
    remaining_indices = []
    text_to_candidates = {entry['Cleaned_Text']: [entry] for entry in base_data}
    
    total_phases = 3
    current_phase = 0
    
    # Phase 1: Exact string matching
    if progress_callback:
        progress_callback(current_phase, total_phases, "🔍 Checking for exact matches...")
    
    for i, user_entry in enumerate(user_data):
        user_text = user_entry['Cleaned_Text']
        if user_text in text_to_candidates:
            chosen = choose_by_hierarchy(user_entry, text_to_candidates[user_text])
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

    # Log exact match results
    exact_count = len([i for i in exact_matches if exact_matches[i] and exact_matches[i][0]['Similarity_Score'] == 1.0])
    logger.info(f"Phase 1 complete: {exact_count} exact matches found, {len(remaining_user_data)} queries need embedding search")

    # Phase 2: Embedding search
    current_phase += 1
    if remaining_user_data:
        if progress_callback:
            progress_callback(current_phase, total_phases, f"🔍 Processing embeddings ({len(remaining_user_data)} queries)...")
        
        logger.info(f"Processing {len(remaining_user_data)} queries with embeddings")
        remaining_texts = [e['Cleaned_Text'] for e in remaining_user_data]
        user_embeddings = model.encode(remaining_texts, convert_to_numpy=True)
        
        D, I = index.search(user_embeddings, top_k)
        
        # Process search results
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
                # Add metadata
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
            progress_callback(current_phase, total_phases, "🔍 Embedding search skipped (all exact matches)")

    # Phase 3: Collect results
    current_phase += 1
    if progress_callback:
        progress_callback(current_phase, total_phases, "🔍 Collecting and finalizing results...")

    # Collect results
    for i in range(len(user_data)):
        if i in exact_matches:
            results.extend(exact_matches[i])

    # Create full embeddings array (for compatibility)
    full_user_embeddings = np.zeros((len(user_data), model.get_sentence_embedding_dimension()))
    if remaining_user_data:
        remaining_embeddings = model.encode([entry['Cleaned_Text'] for entry in remaining_user_data], 
                                           convert_to_numpy=True, show_progress_bar=False)
        for idx, remaining_idx in enumerate(remaining_indices):
            full_user_embeddings[remaining_idx] = remaining_embeddings[idx]

    # Final statistics
    embedding_count = len(remaining_user_data)
    logger.info(f"Search complete: {exact_count} exact matches, {embedding_count} via embeddings")
    logger.info(f"API optimization: Saved ~{exact_count} embedding calls by using string match")
    logger.info(f"Total results generated: {len(results)}")

    # Final progress update
    if progress_callback:
        progress_callback(total_phases, total_phases, f"🔍 Search complete! {len(results)} results generated")

    return results, full_user_embeddings