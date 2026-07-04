from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pint import DimensionalityError, UndefinedUnitError, UnitRegistry

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
    return re.sub(r"^[\s\-•*\.\u2022]+", "", text)


def enhanced_text_cleaning(text):
    """Enhanced text cleaning with specific character preservation and multiple whitespace handling."""
    if text is None:
        return None

    # Keep meaningful characters like the similarity assist code
    # Preserves: letters, numbers, common punctuation, brackets, special chars, accented chars
    text = re.sub(r"[^a-zA-Z0-9.,/()\[\]{}<>\-\s_äöüßÄÖÜáéíóúÁÉÍÓÚñÑçÇ]", "", text)

    # Replace multiple types of whitespace characters (from similarity assist)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\xa0", " ")

    # Normalize multiple spaces to single space
    text = re.sub(r"\s+", " ", text)

    # Strip and ensure single trailing space for consistency
    text = text.strip()

    return text


def clean_whitespace(text):
    """Remove extra whitespace (keeping original function for backward compatibility)."""
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_tokens(text, tokenizer, max_tokens=8000):  # noqa: PLR0911
    """Truncate text to 512 tokens, return text and truncation flag."""
    if text is None or not isinstance(text, str):
        logger.warning("Invalid text passed to truncate_tokens: %s", text)
        return "", True

    # Handle case where tokenizer is None (OpenAI embeddings)
    if tokenizer is None:
        logger.debug("Tokenizer is None, using character-based truncation approximation")
        # For OpenAI embeddings, use character-based approximation
        # Roughly 4 characters per token for English text
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            logger.warning(
                f"Text truncated by character count: {text[:50]}... ({len(text)} chars -> {max_chars} chars)"
            )
            return text[:max_chars], True
        return text, False

    try:
        # Check if tokenizer has the encode method
        if not hasattr(tokenizer, "encode"):
            logger.warning("Tokenizer missing encode method, using character-based truncation")
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                logger.warning(
                    f"Text truncated by character count: {text[:50]}... ({len(text)} chars)"
                )
                return text[:max_chars], True
            return text, False

        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_tokens:
            logger.warning("Text truncated: %s... (%s tokens)", text[:50], len(tokens))

            # Check if tokenizer has decode method and sep_token_id
            if hasattr(tokenizer, "decode") and hasattr(tokenizer, "sep_token_id"):
                try:
                    tokens = tokens[: max_tokens - 2] + [tokenizer.sep_token_id]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                except NotImplementedError:
                    # C-1: MockTokenizer signals it can't decode — use character truncation
                    max_chars = (max_tokens - 2) * 4
                    text = text[:max_chars]
            else:
                # Fallback to character-based truncation
                max_chars = (max_tokens - 2) * 4
                text = text[:max_chars]

            return text, True
        return text, False

    except Exception as e:
        logger.error("Error in truncate_tokens: %s, text: %s", e, text)
        # Fallback to character-based truncation
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            logger.warning("Fallback character truncation applied: %s...", text[:50])
            return text[:max_chars], True
        return text, False


def detect_empty_invalid(text, identifier):
    """Check for empty or non-string inputs."""
    if not isinstance(text, str) or not text.strip():
        logger.warning("Invalid/empty text for ID: %s", identifier)
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
    hierarchy_pattern = r"^\s*(\d+(?:[.\-]\d+)*)[.:]?\s*"
    match = re.match(hierarchy_pattern, text)

    if match:
        hierarchy = match.group(1)
        cleaned_text = re.sub(hierarchy_pattern, "", text)
        logger.debug("Extracted hierarchy: '%s' from text: '%s...'", hierarchy, text[:50])
        return hierarchy, cleaned_text
    else:
        return None, text


def remove_hierarchy(text):
    """Remove hierarchy numbers (e.g., 1.2.3). Kept for backward compatibility."""
    if text is None:
        return None
    cleaned = re.sub(r"^\d+(?:[.\-]\d+)*[.:]?\s*", "", text)
    return cleaned


def add_leading_zero(text):
    """Add leading zero to decimal numbers."""
    if text is None:
        return None
    return re.sub(r"(?<!\d)(?<!\.)\.(\d+)", r"0.\1", text)


def normalize_formatting(text: str) -> str:
    """M-5: Normalize surface formatting differences that don't change meaning.

    Applied before normalize_units() so Pint sees clean, unambiguous input.
    Handles: abbreviation dots (r.p.m → rpm), thousand separators (5,000 → 5000),
    Unicode operators (≥ → >=), smart quotes, and extra whitespace.
    """
    if not isinstance(text, str):
        return text

    # Dotted abbreviations: r.p.m → rpm, e.g. → eg
    text = re.sub(r"\b([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.", r"\1\2\3", text)
    text = re.sub(r"\b([a-zA-Z])\.([a-zA-Z])\.", r"\1\2", text)

    # Thousand separators inside numbers: 5,000 → 5000, 1,000,000 → 1000000
    # Loop to handle nested separators (applied left-to-right each pass)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)

    # Unicode comparison operators → ASCII equivalents
    text = text.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=").replace("≈", "~=")

    # Smart / curly quotes → straight ASCII quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def normalize_units(text):
    """Normalize units using Pint."""
    if text is None:
        return None

    try:
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
        unit_pattern = r"(\d+\.?\d*)\s*([a-zA-Z/]+)"
        matches = re.findall(unit_pattern, text)
        for num, unit in matches:
            try:
                quantity = ureg(f"{num} {unit}")
                normalized = f"{quantity.to_base_units():~}"
                text = text.replace(f"{num} {unit}", normalized)
            except (UndefinedUnitError, DimensionalityError):
                logger.warning("Failed to normalize unit: %s %s", num, unit)
                continue
    except Exception as e:
        logger.error("Unit normalization error: %s", str(e))
    return text


def preprocess_sentence(  # noqa: PLR0911, PLR0912
    entry: dict[str, Any],
    tokenizer: Any | None,
    use_enhanced_cleaning: bool = True,
) -> dict[str, Any] | None:
    """
    Preprocess a single sentence with enhanced cleaning options.
    Now extracts and preserves hierarchy information as metadata.

    Args:
        entry: Dictionary containing text data.
        tokenizer: Tokenizer for text processing (can be None for OpenAI embeddings).
        use_enhanced_cleaning: Whether to use enhanced text cleaning.

    Returns:
        Preprocessed metadata dict, or None if the entry should be skipped.
    """
    identifier = entry.get("Object_Identifier", "Unknown")
    original_text = entry.get("Object_Text", "")

    text = detect_empty_invalid(original_text, identifier)
    if text is None:
        logger.warning("Skipping entry with ID %s due to invalid text", identifier)
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
        logger.debug("Saved hierarchy '%s' for ID: %s", hierarchy, identifier)
    else:
        metadata["Hierarchy"] = None

    # Preprocessing pipeline with enhanced cleaning option and None checks
    text = add_leading_zero(text)
    if text is None:
        logger.warning("Text became None after add_leading_zero for ID: %s", identifier)
        return None

    # M-5: normalize surface formatting before unit parsing so Pint sees clean input
    text = normalize_formatting(text)

    text = normalize_units(text)
    if text is None:
        logger.warning("Text became None after normalize_units for ID: %s", identifier)
        return None

    # Choose between enhanced or standard cleaning
    if use_enhanced_cleaning:
        text = enhanced_text_cleaning(text)  # New enhanced cleaning
    else:
        text = clean_whitespace(text)  # Original cleaning

    if text is None:
        logger.warning("Text became None after cleaning for ID: %s", identifier)
        return None

    text = lowercase_text(text)
    if text is None:
        logger.warning("Text became None after lowercase for ID: %s", identifier)
        return None

    text = remove_bullet_points(text)
    if text is None:
        logger.warning("Text became None after remove_bullet_points for ID: %s", identifier)
        return None

    # Additional check before tokenization
    if not text or not text.strip():
        logger.warning("Text became empty after preprocessing for ID: %s", identifier)
        return None

    text, is_truncated = truncate_tokens(text, tokenizer)
    metadata["Cleaned_Text"] = text
    metadata["Truncated"] = is_truncated

    return metadata


def preprocess_data(
    data: list[dict[str, Any]],
    tokenizer: Any | None,
    use_enhanced_cleaning: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """
    Preprocess a list of sentences with enhanced cleaning options.

    Args:
        data: List of entries to process.
        tokenizer: Tokenizer for text processing (can be None for OpenAI embeddings).
        use_enhanced_cleaning: Whether to use enhanced text cleaning.

    Returns:
        Tuple of (processed_results, skipped_count).
    """
    results = []
    skipped_empty_count = 0

    for entry in data:
        try:
            processed = preprocess_sentence(entry, tokenizer, use_enhanced_cleaning)
            if processed:
                if not processed["Cleaned_Text"] or not processed["Cleaned_Text"].strip():
                    logger.warning("Empty Cleaned_Text for ID: %s", processed["Object_Identifier"])
                    skipped_empty_count += 1
                    continue
                results.append(processed)
            else:
                skipped_empty_count += 1
        except Exception as e:
            logger.error(
                "Error processing entry %s: %s", entry.get("Object_Identifier", "Unknown"), e
            )
            skipped_empty_count += 1
            continue

    logger.info("Processed %s entries, skipped %s empty entries", len(results), skipped_empty_count)

    # Log hierarchy extraction statistics
    hierarchy_count = sum(1 for entry in results if entry.get("Hierarchy"))
    logger.info("Extracted hierarchy information from %s/%s entries", hierarchy_count, len(results))

    return results, skipped_empty_count


# ---------------------------------------------------------------------------
# Section-Aware Hierarchical Comparison — Skill 1 & 2
# ---------------------------------------------------------------------------


@dataclass
class SectionStrategy:
    """Describes how to detect section headings in a requirements spreadsheet."""

    strategy: str = "no_structure"
    heading_column: str | None = None
    heading_values: list = field(default_factory=list)
    number_column: str | None = None
    depth_threshold: int | None = None
    text_column: str | None = None
    confidence: float = 0.0
    reason: str = ""


# Module-level in-process cache (hash → SectionStrategy)
_STRATEGY_CACHE: dict[str, SectionStrategy] = {}


def detect_section_strategy(
    df: pd.DataFrame,
    file_hash: str | None = None,
) -> SectionStrategy:
    """Skill 1 — Detect the section structure strategy of a requirements DataFrame.

    Uses a simple rule-based heuristic first; only calls the LLM when the
    heuristic is inconclusive.  Results are cached in-process by file_hash.

    Args:
        df:        The full DataFrame (before preprocessing).
        file_hash: Optional SHA-256 hex string to cache the result.

    Returns:
        A ``SectionStrategy`` instance.
    """
    if file_hash and file_hash in _STRATEGY_CACHE:
        logger.debug("Section strategy cache hit for hash %s", file_hash[:8])
        return _STRATEGY_CACHE[file_hash]

    from am_ais_assist.config import MAX_SAMPLE_ROWS_FOR_DETECTION

    strategy = _heuristic_section_strategy(df)

    if strategy is None:
        # Fall back to LLM-based detection (local import — constraint #3)
        sample_rows = (
            df.head(MAX_SAMPLE_ROWS_FOR_DETECTION).fillna("").astype(str).to_dict("records")
        )
        from am_ais_assist.llm_service import call_section_detector

        raw = call_section_detector(list(df.columns), sample_rows)
        strategy = SectionStrategy(
            strategy=raw.get("strategy", "no_structure"),
            heading_column=raw.get("heading_column"),
            heading_values=raw.get("heading_values") or [],
            number_column=raw.get("number_column"),
            depth_threshold=raw.get("depth_threshold"),
            text_column=raw.get("text_column"),
            confidence=float(raw.get("confidence", 0.0)),
            reason=raw.get("reason", ""),
        )

    if file_hash:
        _STRATEGY_CACHE[file_hash] = strategy

    logger.info(
        "Section strategy: %s (confidence=%.2f) — %s",
        strategy.strategy,
        strategy.confidence,
        strategy.reason,
    )
    return strategy


def _heuristic_section_strategy(df: pd.DataFrame) -> SectionStrategy | None:
    """Fast rule-based heuristic — returns None when inconclusive."""
    # 1. Look for a dedicated type / heading column
    type_col_candidates = [
        c
        for c in df.columns
        if any(kw in c.lower() for kw in ("type", "kind", "row_type", "object_type", "category"))
    ]
    for col in type_col_candidates:
        values = df[col].dropna().astype(str).unique().tolist()
        heading_vals = [
            v
            for v in values
            if any(h in v.lower() for h in ("heading", "section", "chapter", "title"))
        ]
        if heading_vals:
            return SectionStrategy(
                strategy="object_type_column",
                heading_column=col,
                heading_values=heading_vals,
                confidence=0.95,
                reason=f"Column '{col}' contains heading marker values.",
            )

    # 2. Look for dot-hierarchy numbering in ID-like columns
    id_col_candidates = [
        c
        for c in df.columns
        if any(kw in c.lower() for kw in ("id", "no", "number", "num", "identifier", "ref"))
    ]
    for col in id_col_candidates:
        sample = df[col].dropna().astype(str).head(50)
        dotted = [v for v in sample if re.match(r"^\d+(\.\d+)+$", v.strip())]
        if len(dotted) >= max(3, len(sample) // 4):
            depths = [v.count(".") + 1 for v in dotted]
            max_depth = max(depths)
            if max_depth >= 3:
                return SectionStrategy(
                    strategy="hierarchy_depth",
                    number_column=col,
                    depth_threshold=2,
                    confidence=0.85,
                    reason=f"Column '{col}' has dot-hierarchy up to depth {max_depth}.",
                )

    # Inconclusive — let LLM decide
    return None


def is_section_heading(entry: dict, strategy: SectionStrategy) -> bool:
    """Return True when an entry qualifies as a section heading."""
    # STRATEGY 1: Check for a dedicated "Object Type" column
    if strategy.strategy == "object_type_column" and strategy.heading_column:
        row_type = str(entry.get(strategy.heading_column, "")).strip().lower()
        heading_values_lower = {str(v).lower() for v in strategy.heading_values}
        if row_type in heading_values_lower:
            return True

    # STRATEGY 2: Check the depth of the hierarchy number
    if strategy.strategy == "hierarchy_depth" and strategy.number_column and strategy.depth_threshold:
        # FIX BUG 6: excel_to_json() renames the user-selected id_col to
        # "Object_Identifier". If the strategy was detected on the raw DataFrame
        # before renaming, strategy.number_column holds the original column name
        # which no longer exists in preprocessed entries. We fall back to
        # "Object_Identifier" so depth detection always finds a value.
        # Note: pipeline.py also applies BUG 7 fix (remaps number_column to
        # "Object_Identifier" when it equals the user-selected id_col), so
        # this fallback is a belt-and-suspenders safety net.
        id_val = str(
            entry.get(strategy.number_column) or entry.get("Object_Identifier", "")
        ).strip()
        if not id_val:
            return False
        # Depth is number of dots + 1 (e.g., "1.2" is depth 2)
        depth = id_val.count(".") + 1
        return depth <= strategy.depth_threshold

    # STRATEGY 3: Regex and heuristics on the text itself
    if strategy.strategy == "regex_on_text":
        text = str(entry.get("Original_Text", "")).strip()
        words = text.split()
        requirement_keywords = {"shall", "must", "will", "should", "is required"}

        # Heuristic: Short text that doesn't contain requirement keywords
        if len(words) <= 8 and not any(kw in text.lower() for kw in requirement_keywords):
            # And doesn't end like a sentence
            if not text.endswith((".", ":", ";", "?")):
                return True

        # Heuristic: Text is in ALL CAPS and is not just a single acronym
        if text.isupper() and len(words) >= 2:
            return True

    # Default to False if no strategy matches
    return False


def build_hierarchy_tree(
    processed_data: list[dict],
    strategy: SectionStrategy,
) -> dict[str, list[dict]]:
    """Skill 2 — Build a section_id → [child_entries] mapping.

    Args:
        processed_data: Output of ``preprocess_data()`` / ``excel_to_json()``.
        strategy:       The detected section strategy.

    Returns:
        Dict mapping section heading text to the list of requirement entries
        under that section.  Returns ``{}`` for ``no_structure`` strategy.
    """
    if strategy.strategy == "no_structure":
        return {}

    tree: dict[str, list[dict]] = {}
    current_section: str | None = None

    for entry in processed_data:
        if is_section_heading(entry, strategy):
            current_section = str(
                entry.get("Original_Text", entry.get("Object_Identifier", ""))
            ).strip()
            if current_section not in tree:
                tree[current_section] = []
        else:
            if current_section is not None:
                tree[current_section].append(entry)
            else:
                tree.setdefault("_ungrouped_", []).append(entry)

    # Remove empty sections (headings with no children)
    tree = {k: v for k, v in tree.items() if v}

    logger.info(
        "build_hierarchy_tree: %d sections, %d total items",
        len(tree),
        sum(len(v) for v in tree.values()),
    )
    return tree


# Optional: Batch processing function for large datasets
def preprocess_data_batched(data, tokenizer, batch_size=1000, use_enhanced_cleaning=True):
    """Process data in batches for memory efficiency."""
    results = []
    skipped_empty_count = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        batch_results, batch_skipped = preprocess_data(batch, tokenizer, use_enhanced_cleaning)
        results.extend(batch_results)
        skipped_empty_count += batch_skipped

        logger.info("Processed batch %s/%s", i // batch_size + 1, (len(data) - 1) // batch_size + 1)

    return results, skipped_empty_count
