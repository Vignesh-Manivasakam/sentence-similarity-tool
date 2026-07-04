# Agent-Based Section-Aware Hierarchical Comparison — Skills Reference

This document describes the **6-skill pipeline** that powers the hierarchical requirements comparison in AM AIS Assist.  
The pipeline replaces flat global cosine similarity with a structured, section-scoped comparison that can detect **New Requirements** and **Deleted Requirements** across specification versions.

---

## Overview

```
Input Files (base spec + new spec)
        │
        ▼
  Skill 0 — Preprocessing
        │
        ▼
  Skill 1 — Section Structure Detection
        │
        ▼
  Skill 2 — Hierarchy Tree Building
        │
        ▼
  Skill 3 — Section Mapping (base ↔ new)
        │
        ▼
  Skill 4 — Scoped Requirement Comparison
        │
        ▼
  Skill 5 — Gap Analysis (New / Deleted)
        │
        ▼
   Results (enriched with LLM analysis)
```

---

## Skill 0 — Preprocessing

**Module:** `preprocess.py` (`preprocess_data`, `preprocess_sentence`)  
**Purpose:** Clean and normalise requirement text before embedding or comparison.

### 3-Layer Unit Conversion Hierarchy

| Layer | Operation | Example |
|-------|-----------|---------|
| 1 | Surface formatting normalisation (`normalize_formatting`) | `r.p.m` → `rpm`, `5,000` → `5000`, `≥` → `>=` |
| 2 | Pint unit normalisation (`normalize_units`) | `50 km/h` → `13.889 m/s` |
| 3 | Text cleaning & lowercasing (`enhanced_text_cleaning`, `lowercase_text`) | Remove noise characters, collapse whitespace |

### Preprocessing Pipeline Steps

1. `detect_empty_invalid` — reject blank / non-string inputs.
2. `extract_and_remove_hierarchy` — extract and preserve dot-hierarchy numbers (e.g. `1.2.3`) as metadata.
3. `add_leading_zero` — fix decimal numbers without leading zero.
4. `normalize_formatting` — surface formatting pass (Layer 1).
5. `normalize_units` — Pint unit conversion (Layer 2).
6. `enhanced_text_cleaning` / `clean_whitespace` — noise removal (Layer 3).
7. `lowercase_text` — uniform casing.
8. `remove_bullet_points` — strip leading bullets/dashes.
9. `truncate_tokens` — cap at 512 tokens; uses character-based approximation when no tokenizer.

---

## Skill 1 — Section Structure Detection

**Module:** `preprocess.py` (`detect_section_strategy`, `_heuristic_section_strategy`)  
**LLM call:** `llm_service.call_section_detector()` (Gemini 2.5 Flash Lite, temperature=0.0)  
**Config:** `SECTION_DETECTION_PROMPT_PATH`, `MAX_SAMPLE_ROWS_FOR_DETECTION`

Detects *how* a requirements spreadsheet organises its content into sections.

### Strategies

| Strategy | Detection Rule |
|----------|---------------|
| `object_type_column` | A dedicated column (e.g. "Object Type") contains values like "Heading" or "Section" |
| `hierarchy_depth` | An ID column contains dot-hierarchy numbers (e.g. `1.2.3`); rows with depth ≤ threshold are headings |
| `regex_on_text` | A text column contains ALL-CAPS lines or very short lines without terminal punctuation |
| `no_structure` | No detectable hierarchy; fall back to flat global search |

### SectionStrategy Dataclass

```python
@dataclass
class SectionStrategy:
    strategy: str          # one of the four strategies above
    heading_column: str | None
    heading_values: list   # e.g. ["Heading", "Section"]
    number_column: str | None
    depth_threshold: int | None
    text_column: str | None
    confidence: float      # 0.0–1.0
    reason: str
```

### Caching

Results are cached in-process by SHA-256 file hash (`_STRATEGY_CACHE` dict) so the LLM is called at most once per unique file per process lifetime.

---

## Skill 2 — Hierarchy Tree Building

**Module:** `preprocess.py` (`build_hierarchy_tree`, `is_section_heading`)  
**Config:** (uses `SectionStrategy` from Skill 1)

Organises preprocessed requirements into a `section_title → [entries]` mapping.

### Mapping Rules

1. Iterate through `processed_data` in order.
2. When `is_section_heading(entry, strategy)` returns `True`, open a new section bucket using the entry's `Original_Text` as the key.
3. All subsequent non-heading entries are appended to the current section bucket.
4. Entries before the first heading go into a synthetic `"_ungrouped_"` bucket.
5. Empty sections (headings with no children) are pruned from the result.
6. Returns `{}` when strategy is `no_structure` → triggers flat search fallback.

---

## Skill 3 — Section Mapping

**Module:** `core.py` (`match_sections`)  
**LLM call:** `llm_service.call_agent_decision()` for borderline pairs  
**Config:** `SECTION_AUTO_MATCH_THRESHOLD = 0.90`, `SECTION_UNCERTAIN_THRESHOLD = 0.70`

Maps each base specification section to its counterpart in the new specification.

### Matching Algorithm

1. Embed all section titles (base + new) using the OpenAI embedding model.
2. Compute cosine similarity matrix (`base_titles × new_titles`).
3. **Greedy best-match pass** (sorted by descending score):
   - Score ≥ `SECTION_AUTO_MATCH_THRESHOLD` → auto-accept.
   - `SECTION_UNCERTAIN_THRESHOLD` ≤ score < `SECTION_AUTO_MATCH_THRESHOLD` → verify with Gemini YES/NO.
   - Score < `SECTION_UNCERTAIN_THRESHOLD` → no match.
4. Unmatched base sections → `(base_section, None)` pairs (all items → Deleted).
5. Unmatched new sections → `(None, new_section)` pairs (all items → New Requirement).

---

## Skill 4 — Scoped Requirement Comparison

**Module:** `core.py` (`scoped_search`)  
**LLM call:** `llm_service.call_agent_decision()` for borderline item matches  
**Config:** `ITEM_NEW_REQ_THRESHOLD = 0.50`, `DEFAULT_THRESHOLDS`

Compares requirements within each matched section pair.

### Three-Phase Algorithm

| Phase | Description |
|-------|-------------|
| **A — Exact Match** | Cleaned-text string equality → score = 1.0, label = "Exact Match" |
| **B — Embedding Search** | Local FAISS `IndexFlatIP` over base section; score < `ITEM_NEW_REQ_THRESHOLD` → New Requirement; borderline `[moderate, moderate+0.15]` → Gemini verification |
| **C — Deleted Detection** | Any base item not claimed by any new item → `build_deleted_row()` |

---

## Skill 5 — Gap Analysis

**Module:** `core.py` (`build_new_requirement_row`, `build_deleted_row`)  
**Config:** `PLACEHOLDER_NO_MATCH`, `PLACEHOLDER_DELETED`, `PLACEHOLDER_DELETED_QUERY`

Produces standardised result rows for requirements that exist only in one specification version.

### Output Schema

Both functions produce the standard 8-column result schema (`Query_*`, `Matched_*`, `Similarity_Score`, `Similarity_Level`, `Remark`) with these special values:

| Row Type | `Similarity_Level` | `Similarity_Score` | `Matched_Sentence` / `Query_Sentence` |
|----------|-------------------|-------------------|---------------------------------------|
| New Requirement | `"New Requirement"` | `0.0` | `PLACEHOLDER_NO_MATCH` |
| Deleted | `"Deleted"` | `0.0` | `PLACEHOLDER_DELETED_QUERY` |

### Display Colours

| Label | Colour | Hex |
|-------|--------|-----|
| New Requirement | Blue | `#1565C0` |
| Deleted | Purple | `#7B1FA2` |

---

## Fallback Behaviour

The hierarchical pipeline falls back to the **legacy flat `search_similar()` global cosine search** in any of these cases:

- The raw DataFrame cannot be read (corrupted file, unsupported format).
- `detect_section_strategy()` returns `no_structure`.
- `build_hierarchy_tree()` returns an empty dict for either file.
- Any unhandled exception in the hierarchical path (logged as WARNING).

Ungrouped items (entries not captured by any detected section) are always processed via flat search as a safety net.

---

## Configuration Constants (`config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `SECTION_AUTO_MATCH_THRESHOLD` | `0.90` | Auto-accept section pair above this cosine score |
| `SECTION_UNCERTAIN_THRESHOLD` | `0.70` | Verify with Gemini between this and auto-accept |
| `ITEM_NEW_REQ_THRESHOLD` | `0.50` | Below this score → New Requirement |
| `MAX_SAMPLE_ROWS_FOR_DETECTION` | `10` | Rows sent to Gemini for structure detection |
| `SECTION_DETECTION_PROMPT_PATH` | `prompts/section_detection_prompt.txt` | Prompt for structure detection |
| `AGENT_DECISION_PROMPT_PATH` | `prompts/agent_decision_prompt.txt` | Prompt for binary YES/NO decisions |
| `PLACEHOLDER_NO_MATCH` | `"No equivalent found in base specification"` | Matched_Sentence for New Requirements |
| `PLACEHOLDER_DELETED` | `"Not present in new specification"` | Remark for deleted items |
| `PLACEHOLDER_DELETED_QUERY` | `"(Absent in new file — present in base)"` | Query_Sentence for deleted items |
