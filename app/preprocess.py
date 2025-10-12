import re
import logging
from pint import UnitRegistry, UndefinedUnitError, DimensionalityError

logger = logging.getLogger(__name__)
ureg = UnitRegistry()

def lowercase_text(text):
    """Convert text to lowercase."""
    if text is None:
        return None
    return text.lower()

def remove_bullet_points(text):
    """Remove bullet points or dashes at the start of the text."""
    if text is None:
        return None
    return re.sub(r'^[\s\-•*\.\u2022]+', '', text)

def enhanced_text_cleaning(text):
    """Enhanced text cleaning with specific character preservation and multiple whitespace handling."""
    if text is None:
        return None
    
    # Keep meaningful characters like the similarity assist code
    # Preserves: letters, numbers, common punctuation, brackets, special chars, accented chars
    text = re.sub(r'[^a-zA-Z0-9.,/()\[\]{}<>\-\s_äöüßÄÖÜáéíóúÁÉÍÓÚñÑçÇ]', '', text)
    
    # Replace multiple types of whitespace characters (from similarity assist)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\xa0', ' ')
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip and ensure single trailing space for consistency
    text = text.strip()
    
    return text

def clean_whitespace(text):
    """Remove extra whitespace (keeping original function for backward compatibility)."""
    if text is None:
        return None
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def truncate_tokens(text, tokenizer, max_tokens=8000):
    """Truncate text to 512 tokens, return text and truncation flag."""
    if text is None or not isinstance(text, str):
        logger.warning(f"Invalid text passed to truncate_tokens: {text}")
        return "", True
    
    # Handle case where tokenizer is None (OpenAI embeddings)
    if tokenizer is None:
        logger.debug("Tokenizer is None, using character-based truncation approximation")
        # For OpenAI embeddings, use character-based approximation
        # Roughly 4 characters per token for English text
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            logger.warning(f"Text truncated by character count: {text[:50]}... ({len(text)} chars -> {max_chars} chars)")
            return text[:max_chars], True
        return text, False
    
    try:
        # Check if tokenizer has the encode method
        if not hasattr(tokenizer, 'encode'):
            logger.warning("Tokenizer missing encode method, using character-based truncation")
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                logger.warning(f"Text truncated by character count: {text[:50]}... ({len(text)} chars)")
                return text[:max_chars], True
            return text, False
        
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_tokens:
            logger.warning(f"Text truncated: {text[:50]}... ({len(tokens)} tokens)")
            
            # Check if tokenizer has decode method and sep_token_id
            if hasattr(tokenizer, 'decode') and hasattr(tokenizer, 'sep_token_id'):
                tokens = tokens[:max_tokens-2] + [tokenizer.sep_token_id]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                # Fallback to character-based truncation
                max_chars = (max_tokens - 2) * 4
                text = text[:max_chars]
                
            return text, True
        return text, False
        
    except Exception as e:
        logger.error(f"Error in truncate_tokens: {e}, text: {text}")
        # Fallback to character-based truncation
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            logger.warning(f"Fallback character truncation applied: {text[:50]}...")
            return text[:max_chars], True
        return text, False

def detect_empty_invalid(text, identifier):
    """Check for empty or non-string inputs."""
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid/empty text for ID: {identifier}")
        return None
    return text

def extract_and_remove_hierarchy(text):
    """
    Extract hierarchy numbers (e.g., 1.2.3) and remove them from text.
    Returns tuple of (hierarchy_string, cleaned_text).
    """
    if text is None:
        return None, None
    
    # Pattern to match hierarchy at the beginning of text
    hierarchy_pattern = r'^\s*(\d+(?:[.\-]\d+)*)[.:]?\s*'
    match = re.match(hierarchy_pattern, text)
    
    if match:
        hierarchy = match.group(1)
        cleaned_text = re.sub(hierarchy_pattern, '', text)
        logger.debug(f"Extracted hierarchy: '{hierarchy}' from text: '{text[:50]}...'")
        return hierarchy, cleaned_text
    else:
        return None, text

def remove_hierarchy(text):
    """Remove hierarchy numbers (e.g., 1.2.3). Kept for backward compatibility."""
    if text is None:
        return None
    cleaned = re.sub(r'^\d+(?:[.\-]\d+)*[.:]?\s*', '', text)
    return cleaned

def add_leading_zero(text):
    """Add leading zero to decimal numbers."""
    if text is None:
        return None
    return re.sub(r'(?<!\d)(?<!\.)\.(\d+)', r'0.\1', text)

def normalize_units(text):
    """Normalize units using Pint."""
    if text is None:
        return None
    
    try:
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        unit_pattern = r'(\d+\.?\d*)\s*([a-zA-Z/]+)'
        matches = re.findall(unit_pattern, text)
        for num, unit in matches:
            try:
                quantity = ureg(f"{num} {unit}")
                normalized = f"{quantity.to_base_units():~}"
                text = text.replace(f"{num} {unit}", normalized)
            except (UndefinedUnitError, DimensionalityError):
                logger.warning(f"Failed to normalize unit: {num} {unit}")
                continue
    except Exception as e:
        logger.error(f"Unit normalization error: {str(e)}")
    return text

def preprocess_sentence(entry, tokenizer, use_enhanced_cleaning=True):
    """
    Preprocess a single sentence with enhanced cleaning options.
    Now extracts and preserves hierarchy information as metadata.
    
    Args:
        entry: Dictionary containing text data
        tokenizer: Tokenizer for text processing (can be None for OpenAI embeddings)
        use_enhanced_cleaning: Whether to use enhanced text cleaning
    """
    identifier = entry.get("Object_Identifier", "Unknown")
    original_text = entry.get("Object_Text", "")
    
    text = detect_empty_invalid(original_text, identifier)
    if text is None:
        logger.warning(f"Skipping entry with ID {identifier} due to invalid text")
        return None
    
    # Start with standard metadata and carry over any additional columns
    metadata = {"Object_Identifier": identifier, "Original_Text": original_text}
    for key, value in entry.items():
        if key not in ["Object_Identifier", "Object_Text"]:
            metadata[key] = value

    # Extract hierarchy before any other processing
    hierarchy, text = extract_and_remove_hierarchy(text)
    if hierarchy:
        metadata["Hierarchy"] = hierarchy
        logger.debug(f"Saved hierarchy '{hierarchy}' for ID: {identifier}")
    else:
        metadata["Hierarchy"] = None

    # Preprocessing pipeline with enhanced cleaning option and None checks
    text = add_leading_zero(text)
    if text is None:
        logger.warning(f"Text became None after add_leading_zero for ID: {identifier}")
        return None
        
    text = normalize_units(text)
    if text is None:
        logger.warning(f"Text became None after normalize_units for ID: {identifier}")
        return None
    
    # Choose between enhanced or standard cleaning
    if use_enhanced_cleaning:
        text = enhanced_text_cleaning(text)  # New enhanced cleaning
    else:
        text = clean_whitespace(text)  # Original cleaning
    
    if text is None:
        logger.warning(f"Text became None after cleaning for ID: {identifier}")
        return None
        
    text = lowercase_text(text)
    if text is None:
        logger.warning(f"Text became None after lowercase for ID: {identifier}")
        return None
        
    text = remove_bullet_points(text)
    if text is None:
        logger.warning(f"Text became None after remove_bullet_points for ID: {identifier}")
        return None
    
    # Additional check before tokenization
    if not text or not text.strip():
        logger.warning(f"Text became empty after preprocessing for ID: {identifier}")
        return None
    
    text, is_truncated = truncate_tokens(text, tokenizer)
    metadata["Cleaned_Text"] = text
    metadata["Truncated"] = is_truncated
    
    return metadata

def preprocess_data(data, tokenizer, use_enhanced_cleaning=True):
    """
    Preprocess a list of sentences with enhanced cleaning options.
    
    Args:
        data: List of entries to process
        tokenizer: Tokenizer for text processing (can be None for OpenAI embeddings)
        use_enhanced_cleaning: Whether to use enhanced text cleaning
    
    Returns:
        Tuple of (processed_results, skipped_count)
    """
    results = []
    skipped_empty_count = 0
    
    for entry in data:
        try:
            processed = preprocess_sentence(entry, tokenizer, use_enhanced_cleaning)
            if processed:
                if not processed["Cleaned_Text"] or not processed["Cleaned_Text"].strip():
                    logger.warning(f"Empty Cleaned_Text for ID: {processed['Object_Identifier']}")
                    skipped_empty_count += 1
                    continue
                results.append(processed)
            else:
                skipped_empty_count += 1
        except Exception as e:
            logger.error(f"Error processing entry {entry.get('Object_Identifier', 'Unknown')}: {e}")
            skipped_empty_count += 1
            continue
    
    logger.info(f"Processed {len(results)} entries, skipped {skipped_empty_count} empty entries")
    
    # Log hierarchy extraction statistics
    hierarchy_count = sum(1 for entry in results if entry.get("Hierarchy"))
    logger.info(f"Extracted hierarchy information from {hierarchy_count}/{len(results)} entries")
    
    return results, skipped_empty_count

# Optional: Batch processing function for large datasets
def preprocess_data_batched(data, tokenizer, batch_size=1000, use_enhanced_cleaning=True):
    """Process data in batches for memory efficiency."""
    results = []
    skipped_empty_count = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_results, batch_skipped = preprocess_data(batch, tokenizer, use_enhanced_cleaning)
        results.extend(batch_results)
        skipped_empty_count += batch_skipped
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
    
    return results, skipped_empty_count