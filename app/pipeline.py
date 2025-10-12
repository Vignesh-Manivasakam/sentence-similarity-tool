import os
import logging
import json
import pandas as pd
import streamlit as st
from app.utils import excel_to_json
from app.config import OUTPUT_DIR, LLM_ANALYSIS_MIN_THRESHOLD, LLM_PERFECT_MATCH_THRESHOLD
from app.core import load_model as core_load_model
from app.core import create_faiss_index, search_similar
from app.llm_service import get_llm_analysis_batch
from app.progress_manager import UnifiedProgressManager

@st.cache_resource
def get_model():
    """A Streamlit-cached wrapper to load the model only once."""
    logging.info("get_model() called. Caching model resource.")
    return core_load_model()

# Load the model once using the cached function
model = get_model()
tokenizer = model.tokenizer


def get_user_output_dir(user_session_id):
    """Get user-specific output directory"""
    if user_session_id:
        from app.config import get_user_output_dir as config_get_user_output_dir
        return config_get_user_output_dir(user_session_id)
    else:
        return OUTPUT_DIR


def run_llm_analysis_phase(faiss_results, user_session_id=None, progress_callback=None):
    """
    Run LLM analysis on FAISS results.
    
    Args:
        faiss_results: List of FAISS search results
        user_session_id: User session for caching
        progress_callback: Callback function for progress updates (callback(current, total, message))
    
    Returns:
        List of enriched results with LLM analysis
    """
    if not faiss_results:
        logging.warning("No FAISS results provided for LLM analysis")
        return faiss_results
    
    try:
        logging.info(f"Starting LLM analysis for {len(faiss_results)} results")
        
        # Convert to DataFrame for easier processing
        initial_df = pd.DataFrame(faiss_results)
        llm_results = []
        total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}
        total_rows = len(initial_df)
        batch_size = 10
        
        # Calculate total batches for progress tracking
        total_batches = -(-total_rows // batch_size)  # Ceiling division
        
        if progress_callback:
            progress_callback(0, total_batches, "Starting LLM analysis...")

        for batch_idx in range(0, total_rows, batch_size):
            batch_rows = initial_df.iloc[batch_idx:batch_idx + batch_size]
            sentence_pairs = [
                (row['Query_Sentence_Cleaned_text'], row['Matched_Sentence_Cleaned_text'])
                for _, row in batch_rows.iterrows()
            ]
            batch_results = [None] * len(sentence_pairs)
            llm_pairs = []
            llm_indices = []

            # Process each pair in the batch
            for j, (sentence1, sentence2) in enumerate(sentence_pairs):
                score = batch_rows.iloc[j]['Similarity_Score']
                
                # Skip LLM analysis for perfect matches and low similarity scores
                if score >= LLM_PERFECT_MATCH_THRESHOLD:
                    batch_results[j] = {
                        'Similarity_Score': 1.0,
                        'Similarity_Level': 'Exact Match',
                        'Remark': 'Exactly Matched'
                    }
                elif score < LLM_ANALYSIS_MIN_THRESHOLD:
                    batch_results[j] = {
                        'Similarity_Score': 'N/A',
                        'Similarity_Level': 'Below Threshold',
                        'Remark': 'Analysis skipped as similarity score is below threshold.'
                    }
                else:
                    llm_pairs.append((sentence1, sentence2))
                    llm_indices.append(j)

            # Call LLM for remaining pairs that need analysis
            if llm_pairs:
                try:
                    batch_response = get_llm_analysis_batch(llm_pairs, user_session_id)
                    llm_batch_results = batch_response['results']
                    total_tokens['prompt_tokens'] += batch_response['tokens_used']['prompt_tokens']
                    total_tokens['completion_tokens'] += batch_response['tokens_used']['completion_tokens']

                    for idx, result in zip(llm_indices, llm_batch_results):
                        batch_results[idx] = result
                
                except Exception as e:
                    logging.error(f"LLM analysis failed for batch {batch_idx//batch_size + 1}: {e}")
                    # Fill with error results
                    error_result = {
                        'Similarity_Score': 'Error',
                        'Similarity_Level': 'Analysis Error',
                        'Remark': f'LLM analysis failed: {str(e)}'
                    }
                    for idx in llm_indices:
                        batch_results[idx] = error_result

            llm_results.extend(batch_results)
            
            # Update progress
            current_batch = batch_idx // batch_size + 1
            progress = current_batch / total_batches
            message = f"LLM Analysis... {current_batch}/{total_batches}"
            
            if progress_callback:
                progress_callback(current_batch, total_batches, message)
            
            logging.info(f"Completed LLM batch {current_batch}/{total_batches}")

        # Combine FAISS results with LLM results
        llm_df = pd.DataFrame(llm_results, index=initial_df.index)
        enriched_df = pd.concat([initial_df, llm_df], axis=1)
        
        logging.info(f"LLM analysis complete. Token usage: {total_tokens}")
        
        # Convert back to list of dictionaries and add token info
        enriched_results = enriched_df.to_dict('records')
        
        # Add token usage metadata to the first result for tracking
        if enriched_results:
            enriched_results[0]['_llm_tokens_used'] = total_tokens
        
        return enriched_results, total_tokens
        
    except Exception as e:
        logging.error(f"Error in LLM analysis phase: {e}", exc_info=True)
        # Return original results without LLM analysis
        return faiss_results, {'prompt_tokens': 0, 'completion_tokens': 0}


# In pipeline.py, update the function signature and make thresholds optional:

def run_similarity_pipeline(base_file, check_file, top_k, thresholds, progress_manager, 
                           base_id_col, base_text_col, check_id_col, check_text_col, 
                           base_meta_cols, check_meta_cols, user_session_id=None):
    """
    Run the end-to-end similarity search pipeline with unified progress tracking.
    """
    try:
        session_info = f"session {user_session_id}" if user_session_id else "global session"
        logging.info(f"Starting integrated pipeline for {session_info}")
        
        # ========== PHASE 1: PREPROCESSING ==========
        progress_manager.start_phase('preprocessing', "📄 Processing Excel files...")
        
        base_data, base_skipped = excel_to_json(base_file, tokenizer, base_id_col, base_text_col, base_meta_cols)
        progress_manager.update_phase_progress(0.5, "📄 Processing base file...")
        
        user_data, user_skipped = excel_to_json(check_file, tokenizer, check_id_col, check_text_col, check_meta_cols)
        progress_manager.update_phase_progress(1.0, "📄 File processing complete...")
        
        if not base_data or not user_data:
            raise ValueError("No valid data after preprocessing")
        
        # Save preprocessed data...
        progress_manager.complete_phase("📄 Preprocessing complete!")
        
        # ========== PHASE 2: FAISS INDEX ==========
        progress_manager.start_phase('faiss_index', f"🏗️ Building FAISS index ({len(base_data)} items)...")

        # ✅ UPDATED: Better progress callback for FAISS index creation
        def faiss_index_progress(current_step, total_steps, message):
            progress = current_step / total_steps if total_steps > 0 else 0
            progress_manager.update_phase_progress(progress, message)  # ← Remove the 🏗️ prefix since message already has emoji

        index, base_embeddings, base_data = create_faiss_index(
            base_data, model, base_file, user_session_id, progress_callback=faiss_index_progress
        )

        progress_manager.complete_phase("🏗️ FAISS index ready!")
        
        # ========== PHASE 3: FAISS SEARCH ==========
        progress_manager.start_phase('faiss_search', f"🔍 Searching similarities...")
        
        from app.config import DEFAULT_THRESHOLDS
        internal_thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        
        # Add progress callback for search
        def search_progress(current, total, message):
            progress = current / total if total > 0 else 0
            progress_manager.update_phase_progress(progress, f"🔍 {message}")
        
        faiss_results, user_embeddings = search_similar(
            user_data, index, base_data, top_k, internal_thresholds, model, 
            progress_callback=search_progress
        )
        
        progress_manager.complete_phase(f"🔍 Found {len(faiss_results)} matches!")
        
        # ========== PHASE 4: LLM ANALYSIS ==========
        progress_manager.start_phase('llm_analysis', "🧠 Running LLM analysis...")
        
        def llm_progress_callback(current, total, message):
            progress = current / total if total > 0 else 0
            progress_manager.update_phase_progress(progress, f"🧠 {message}")
        
        try:
            enriched_results, llm_tokens = run_llm_analysis_phase(
                faiss_results, 
                user_session_id, 
                progress_callback=llm_progress_callback
            )
            progress_manager.complete_phase(f"🧠 LLM analysis complete! Tokens: {llm_tokens}")
            
        except Exception as e:
            logging.error(f"LLM analysis failed: {e}")
            progress_manager.update_phase_progress(1.0, f"⚠️ LLM analysis failed, using FAISS only")
            enriched_results = faiss_results
            llm_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}

        # ========== PIPELINE COMPLETE ==========
        progress_manager.complete_all()
        logging.info(f"Integrated pipeline completed successfully for {session_info}")
        
        return enriched_results, base_embeddings, user_embeddings, base_data, user_data, base_skipped, user_skipped, llm_tokens
        
    except Exception as e:
        error_msg = f"Pipeline error for {session_info if 'session_info' in locals() else 'unknown session'}: {e}"
        logging.error(error_msg, exc_info=True)
        if 'progress_manager' in locals():
            progress_manager.progress_bar.progress(0.0, text=f"❌ {error_msg}")
        raise