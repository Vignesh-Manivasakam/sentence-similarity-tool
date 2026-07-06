# 🔍 AI Requirement Similarity Assistant (AIS Assist)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-orange.svg)](https://www.trychroma.com/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-llama--3.1--70b--instruct-green.svg)](https://build.nvidia.com/)
[![Status](https://img.shields.io/badge/Status-Public_POC-success.svg)](#)

An intelligent requirements comparison and compliance engineering workbench. It automates compliance reviews between incoming customer specifications and legacy engineering documents by integrating **ChromaDB** persistent vector storage, **NVIDIA NIM** endpoints, and a **5-Gate Self-Improving Prompt Compiler** with canary deployment routing.

---

## 🏗️ 4-Phase Pipeline

```
Phase 1: Preprocess (Excel→JSON, text cleaning, unit normalization via Pint)
Phase 2: Vector Index (Build/load ChromaDB collection, FAISS IndexFlatIP in memory)
Phase 3: Similarity Search (Hierarchical section-aware matching OR flat FAISS fallback)
Phase 4: LLM Analysis (Feedback recall → Canary routing → Batch semantic enrichment)
```

### Dual-Path Token Saver
1. **Exact String Match** — bypasses embeddings and LLM calls entirely for identical rows (zero cost)
2. **Smart LLM Skipping** — scores ≥ 0.999 auto-labeled "Exact Match", scores < 0.4 auto-labeled "Below Threshold"
3. **Level 1 Learning** — checks each pair against user's ChromaDB feedback collection (cosine ≥ 0.97); reuses stored verdict if found

### Hierarchical Section-Aware Comparison
- **Section detection** — 4 strategies: `object_type_column`, `hierarchy_depth`, `regex_on_text`, `no_structure` (heuristic-first with LLM fallback)
- **Section mapping** — cosine similarity matrix between base and new sections; auto-match ≥ 0.90, borderline [0.70, 0.90) verified with LLM YES/NO call
- **Scoped search** — per-section FAISS comparison with exact match phase, batch-precomputed embeddings, and deleted/new requirement detection

---

## 🔄 5-Gate Self-Improving Prompt Compiler

| Gate | Name | Method |
|------|------|--------|
| **Gate 1** | Statistical Pre-Analysis | Aggregates "Not OK" verdicts by AI level, score range, prompt version. Requires ≥50 verdicts. Pure Python, no LLM. Computes deterministic 20% holdout set. |
| **Gate 2** | LLM Pattern Analysis | Sends ONLY aggregated statistics (never raw text) to LLM. Produces ONE ≤3-sentence prompt addition. Rejects if confidence < 0.60. |
| **Gate 3** | Automated Validation | 4 checks: stat backing (±20pp tolerance), contradiction detection, shadow test on holdout set, confidence threshold. All must pass. |
| **Gate 4** | Human Review | Admin reviews the suggested prompt patch in the Streamlit UI. |
| **Gate 5** | Canary Deployment | Deterministic 10% session routing. After 20 sessions, auto-promotes (≥3pp improvement) or rolls back (<-2pp regression). |

---

## 🧠 Additional Intelligence Layers

* **Per-user skill learning** — extracts matching preferences from user corrections via LLM, persists in SQLite, injects top-10 active skills (confidence ≥ 0.70) into prompt per session
* **Canary prompt registry** — versioned prompt files with filelock-protected JSON registry; deterministic `hash(session_id) % 100` routing; rolling agreement rate tracking
* **Text preprocessing** — unit normalization via **Pint** library (kN→N, etc.), dot abbreviation normalization (r.p.m→rpm), Unicode operators (≥→>=), hierarchy number extraction, 8000-token truncation with tiktoken fallback
* **Security** — gateway secret validation for header spoofing protection, HTML sanitization via bleach allow-list, no raw text in self-improvement LLM calls, 200MB file size limit, 20-user concurrency semaphore

---

## 📁 Repository Structure

```text
sentence-similarity-tool/
├── am_ais_assist/              # Core backend package
│   ├── cache_manager.py        # Thread-safe in-memory LLM result cache
│   ├── config.py               # NVIDIA NIM and ChromaDB configurations
│   ├── core.py                 # ChromaDB client, FAISS IndexFlatIP, section-aware search (1115 lines)
│   ├── feedback_store.py       # Per-user + global ChromaDB feedback persistence (550 lines)
│   ├── llm_service.py          # OpenAI client for NVIDIA NIM LLM endpoints (704 lines)
│   ├── pipeline.py             # 4-phase orchestrator with concurrency control (626 lines)
│   ├── postprocess.py          # HTML rendering, highlighted Excel export, Plotly summaries
│   ├── preprocess.py           # Text cleaning, Pint unit normalization, section detection
│   ├── prompt_registry.py      # Versioned prompts with canary deployment routing (543 lines)
│   ├── self_improve.py         # 5-Gate self-improving prompt compiler (516 lines)
│   ├── skill_generator.py      # Per-user learned matching rules (SQLite + JSON fallback)
│   └── prompts/                # System prompt, pattern analysis, contradiction check, section detection, agent decision
├── app.py                      # Streamlit frontend (978 lines): auth, upload, review, admin panel
├── requirements.txt            # Package dependencies
└── skills.md                   # Antigravity skill definitions
```

---

## 🛠️ Setup & Execution

### 1. Configure Environment
```env
NVIDIA_API_KEY="nvapi-..."
NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

### 2. Install & Launch
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛡️ Corporate Confidentiality Notice
This repository is an anonymized, public proof-of-concept. It does not contain proprietary data or internal intellectual property.
