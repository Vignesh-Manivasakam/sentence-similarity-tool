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

logger = logging.getLogger(__name__)

@st.cache_resource
def get_model():
    """A Streamlit-cached wrapper to load the model only once."""
    logger.info("get_model() called. Caching model resource.")
    return core_load_model()

# Load the model once using the cached function
model = get_model()

# ✅ FIX: Safe tokenizer access - removed direct assignment
# tokenizer is now accessed via model.tokenizer property when needed


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
        Tuple of (enriched_results, token_usage_dict)
    """
    if not faiss_results:
        logger.warning("No FAISS results provided for LLM analysis")
        return faiss_results, {'prompt_tokens': 0, 'completion_tokens': 0}
    
    try:
        logger.info(f"Starting LLM analysis for {len(faiss_results)} results")
        
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
                    total_tokens['prompt_tokens'] += batch_response['tokens_used'].get('prompt_tokens', 0)
                    total_tokens['completion_tokens'] += batch_response['tokens_used'].get('completion_tokens', 0)

                    for idx, result in zip(llm_indices, llm_batch_results):
                        batch_results[idx] = result
                
                except Exception as e:
                    logger.error(f"LLM analysis failed for batch {batch_idx//batch_size + 1}: {e}")
                    # Fill with error results
                    error_result = {
                        'Similarity_Score': 'Error',
                        'Similarity_Level': 'Analysis Error',
                        'Remark': f'LLM analysis failed: {str(e)}'
                    }
                    for idx in llm_indices:
                        if batch_results[idx] is None:
                            batch_results[idx] = error_result

            llm_results.extend(batch_results)
            
            # Update progress
            current_batch = batch_idx // batch_size + 1
            message = f"Analyzing batch {current_batch}/{total_batches}"
            
            if progress_callback:
                progress_callback(current_batch, total_batches, message)
            
            logger.info(f"Completed LLM batch {current_batch}/{total_batches}")

        # Combine FAISS results with LLM results
        llm_df = pd.DataFrame(llm_results, index=initial_df.index)
        
        # Rename LLM columns to avoid conflicts (if FAISS already has these columns)
        llm_df.rename(columns={
            'Similarity_Score': 'LLM_Similarity_Score',
            'Similarity_Level': 'LLM_Similarity_Level',
            'Remark': 'LLM_Remark'
        }, inplace=True)
        
        enriched_df = pd.concat([initial_df, llm_df], axis=1)
        
        logger.info(f"LLM analysis complete. Token usage: {total_tokens}")
        
        # Convert back to list of dictionaries
        enriched_results = enriched_df.to_dict('records')
        
        return enriched_results, total_tokens
        
    except Exception as e:
        logger.error(f"Error in LLM analysis phase: {e}", exc_info=True)
        # Return original results without LLM analysis
        return faiss_results, {'prompt_tokens': 0, 'completion_tokens': 0}


def run_similarity_pipeline(
    base_file, 
    check_file, 
    top_k, 
    thresholds, 
    progress_manager, 
    base_id_col, 
    base_text_col, 
    check_id_col, 
    check_text_col, 
    base_meta_cols, 
    check_meta_cols, 
    user_session_id=None
):
    """
    Run the end-to-end similarity search pipeline with unified progress tracking.
    
    Args:
        base_file: Base file object
        check_file: Check file object
        top_k: Number of top matches to return
        thresholds: Dictionary of similarity thresholds
        progress_manager: UnifiedProgressManager instance
        base_id_col: Column name for base file ID
        base_text_col: Column name for base file text
        check_id_col: Column name for check file ID
        check_text_col: Column name for check file text
        base_meta_cols: List of metadata columns from base file
        check_meta_cols: List of metadata columns from check file
        user_session_id: Optional user session ID for caching
    
    Returns:
        Tuple of (enriched_results, base_embeddings, user_embeddings, base_data, 
                 user_data, base_skipped, user_skipped, llm_tokens)
    """
    try:
        session_info = f"session {user_session_id}" if user_session_id else "global session"
        logger.info(f"Starting integrated pipeline for {session_info}")
        
        # ========== PHASE 1: PREPROCESSING ==========
        progress_manager.start_phase('preprocessing', "📄 Processing Excel files...")
        
        try:
            # ✅ Pass model instead of tokenizer to excel_to_json
            base_data, base_skipped = excel_to_json(
                base_file, model, base_id_col, base_text_col, base_meta_cols
            )
            progress_manager.update_phase_progress(0.5, "📄 Base file processed...")
            
            user_data, user_skipped = excel_to_json(
                check_file, model, check_id_col, check_text_col, check_meta_cols
            )
            progress_manager.update_phase_progress(1.0, "📄 Check file processed...")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise ValueError(f"File processing error: {str(e)}")
        
        if not base_data:
            raise ValueError("No valid data in base file after preprocessing")
        if not user_data:
            raise ValueError("No valid data in check file after preprocessing")
        
        logger.info(f"Preprocessing complete: {len(base_data)} base items, {len(user_data)} check items")
        
        # Save preprocessed data for debugging (optional)
        if user_session_id:
            try:
                output_dir = get_user_output_dir(user_session_id)
                os.makedirs(output_dir, exist_ok=True)
                
                with open(os.path.join(output_dir, 'base_data.json'), 'w', encoding='utf-8') as f:
                    json.dump(base_data[:10], f, indent=2)  # Save sample only
                with open(os.path.join(output_dir, 'user_data.json'), 'w', encoding='utf-8') as f:
                    json.dump(user_data[:10], f, indent=2)  # Save sample only
            except Exception as e:
                logger.warning(f"Could not save preprocessed data samples: {e}")
        
        progress_manager.complete_phase("📄 Preprocessing complete!")
        
        # ========== PHASE 2: FAISS INDEX ==========
        progress_manager.start_phase('faiss_index', f"🏗️ Building FAISS index ({len(base_data)} items)...")

        def faiss_index_progress(current_step, total_steps, message):
            """Progress callback for FAISS index creation"""
            progress = current_step / total_steps if total_steps > 0 else 0
            progress_manager.update_phase_progress(progress, message)

        try:
            index, base_embeddings, base_data = create_faiss_index(
                base_data, model, base_file, user_session_id, progress_callback=faiss_index_progress
            )
            logger.info(f"FAISS index created: {index.ntotal} vectors indexed")
        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create FAISS index: {str(e)}")

        progress_manager.complete_phase("🏗️ FAISS index ready!")
        
        # ========== PHASE 3: FAISS SEARCH ==========
        progress_manager.start_phase('faiss_search', f"🔍 Searching for similarities...")
        
        from app.config import DEFAULT_THRESHOLDS
        internal_thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        
        def search_progress(current, total, message):
            """Progress callback for similarity search"""
            progress = current / total if total > 0 else 0
            progress_manager.update_phase_progress(progress, message)
        
        try:
            faiss_results, user_embeddings = search_similar(
                user_data, index, base_data, top_k, internal_thresholds, model, 
                progress_callback=search_progress
            )
            logger.info(f"FAISS search complete: {len(faiss_results)} matches found")
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            raise RuntimeError(f"Similarity search failed: {str(e)}")
        
        progress_manager.complete_phase(f"🔍 Found {len(faiss_results)} matches!")
        
        # ========== PHASE 4: LLM ANALYSIS ==========
        progress_manager.start_phase('llm_analysis', "🧠 Running LLM analysis...")
        
        def llm_progress_callback(current, total, message):
            """Progress callback for LLM analysis"""
            progress = current / total if total > 0 else 0
            progress_manager.update_phase_progress(progress, message)
        
        try:
            enriched_results, llm_tokens = run_llm_analysis_phase(
                faiss_results, 
                user_session_id, 
                progress_callback=llm_progress_callback
            )
            
            token_msg = f"Prompt: {llm_tokens.get('prompt_tokens', 0):,}, Completion: {llm_tokens.get('completion_tokens', 0):,}"
            progress_manager.complete_phase(f"🧠 LLM analysis complete! {token_msg}")
            logger.info(f"LLM analysis complete. Token usage: {llm_tokens}")
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            progress_manager.update_phase_progress(1.0, f"⚠️ LLM analysis failed, using FAISS results only")
            enriched_results = faiss_results
            llm_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}

        # ========== PIPELINE COMPLETE ==========
        progress_manager.complete_all()
        logger.info(f"✅ Pipeline completed successfully for {session_info}")
        logger.info(f"   - Base items: {len(base_data)} (skipped: {base_skipped})")
        logger.info(f"   - Check items: {len(user_data)} (skipped: {user_skipped})")
        logger.info(f"   - Results: {len(enriched_results)}")
        logger.info(f"   - LLM tokens: {llm_tokens}")
        
        return (
            enriched_results, 
            base_embeddings, 
            user_embeddings, 
            base_data, 
            user_data, 
            base_skipped, 
            user_skipped, 
            llm_tokens
        )
        
    except Exception as e:
        session_info_str = session_info if 'session_info' in locals() else 'unknown session'
        error_msg = f"Pipeline error for {session_info_str}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        if 'progress_manager' in locals():
            try:
                progress_manager.progress_bar.progress(0.0, text=f"❌ {error_msg}")
            except:
                pass
        
        raise RuntimeError(error_msg) from e


def validate_pipeline_inputs(base_file, check_file, base_id_col, base_text_col, check_id_col, check_text_col):
    """
    Validate pipeline inputs before processing.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check files are not None
        if base_file is None:
            return False, "Base file is required"
        if check_file is None:
            return False, "Check file is required"
        
        # Check column names are not empty
        if not base_id_col or not base_text_col:
            return False, "Base file column names are required"
        if not check_id_col or not check_text_col:
            return False, "Check file column names are required"
        
        # Try to read a sample of each file
        try:
            base_df = pd.read_excel(base_file, nrows=5)
            if base_id_col not in base_df.columns:
                return False, f"Column '{base_id_col}' not found in base file"
            if base_text_col not in base_df.columns:
                return False, f"Column '{base_text_col}' not found in base file"
        except Exception as e:
            return False, f"Error reading base file: {str(e)}"
        
        try:
            check_df = pd.read_excel(check_file, nrows=5)
            if check_id_col not in check_df.columns:
                return False, f"Column '{check_id_col}' not found in check file"
            if check_text_col not in check_df.columns:
                return False, f"Column '{check_text_col}' not found in check file"
        except Exception as e:
            return False, f"Error reading check file: {str(e)}"
        
        # Reset file pointers
        base_file.seek(0)
        check_file.seek(0)
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"