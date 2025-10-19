---
title: Sentence Similarity Analysis Tool
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# 📘 AI Similarity Assist Tool 🔍

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Deployed-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**An intelligent dual-analysis system combining FAISS vector similarity with LLM-powered semantic understanding**

[🚀 Live Demo](https://huggingface.co/spaces/Vignesh0503/sentence-similarity-tool) • [📖 Documentation](#features) • [🎯 Use Cases](#use-cases) • [⚡ Quick Start](#quick-start)

</div>

---

## 🌟 Overview

The **AI Similarity Assist Tool** is a production-ready Streamlit application that revolutionizes text similarity analysis by combining two powerful approaches:

1. **⚡ FAISS Vector Search**: Lightning-fast embedding-based similarity using BGE-large-en-v1.5
2. **🧠 LLM Semantic Analysis**: Deep relationship understanding powered by Groq's GPT-OSS-20B

This dual-layer approach ensures both speed and accuracy, making it perfect for document comparison, duplicate detection, semantic search, and content analysis tasks.

---

## ✨ Key Features

### 🎯 **Dual Analysis Engine**
- **FAISS Similarity**: Efficient vector-based matching with cosine similarity scores
- **LLM Analysis**: Contextual relationship classification (Exact Match, Equivalent, Related, Contradictory)
- **Hybrid Results**: Get both quantitative scores and qualitative insights

### 🚀 **Performance Optimized**
- **Smart Caching**: User-specific session management with automatic cleanup
- **Exact Match Detection**: Bypasses embedding for identical strings (saves 50%+ compute time)
- **Batch Processing**: Efficient LLM calls with token-aware batching
- **Progress Tracking**: Real-time unified progress indicators across all phases

### 📊 **Rich Visualizations**
- **3D Embedding Plot**: Interactive Plotly visualization of semantic space
- **Comparative Charts**: Distribution analysis for both FAISS and LLM results
- **Token Usage Metrics**: Track LLM consumption for cost optimization
- **Detailed Statistics**: Comprehensive breakdown by relationship types

### 💾 **Export & Integration**
- **Excel Export**: Highlighted differences with color-coded relationships
- **JSON Export**: Structured data for downstream processing
- **Metadata Support**: Preserve additional columns from source files
- **Truncation Indicators**: Visual warnings for processed long texts

### 🔧 **Enterprise Ready**
- **Multi-User Support**: Isolated sessions for concurrent users (up to 100)
- **Automatic Cleanup**: 24-hour session timeout with garbage collection
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Logging**: Comprehensive tracking for debugging and monitoring

---

## 🎯 Use Cases

| Domain | Application |
|--------|-------------|
| 📄 **Legal & Compliance** | Contract comparison, clause matching, regulatory alignment |
| 🏢 **Enterprise Data** | Duplicate detection, data deduplication, record linkage |
| 📚 **Content Management** | Plagiarism detection, content similarity, version tracking |
| 🎓 **Research & Academia** | Literature review, citation analysis, semantic clustering |
| 🛒 **E-commerce** | Product matching, review analysis, catalog normalization |
| 💬 **Customer Support** | Ticket categorization, FAQ matching, response suggestions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                 │
├─────────────────────────────────────────────────────────────┤
│  File Upload  →  Column Selection  →  Run Analysis          │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │   Pipeline Manager        │
        │   (progress_manager.py)   │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴──────────────────────────────┐
        │                                             │
┌───────▼────────┐                          ┌────────▼─────────┐
│  FAISS Search  │                          │  LLM Analysis    │
│  (core.py)     │                          │  (llm_service.py)│
├────────────────┤                          ├──────────────────┤
│ • BGE-large    │                          │ • Groq GPT-OSS   │
│ • Embeddings   │                          │ • Batch Process  │
│ • Index Cache  │                          │ • Token Tracking │
│ • Exact Match  │                          │ • Result Cache   │
└────────────────┘                          └──────────────────┘
        │                                            │
        └─────────────┬──────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Results Processing      │
        │   (postprocess.py)        │
        ├───────────────────────────┤
        │ • Visualization           │
        │ • Excel/JSON Export       │
        │ • Summary Statistics      │
        └───────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.9+
pip (Python package manager)
```

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Vignesh-Manivasakam/sentence-similarity-tool.git
cd sentence-similarity-tool
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Up Environment Variables**
```bash
# Create .env file or set environment variable
export GROQ_API_KEY="your_groq_api_key_here"
```

4. **Run the Application**
```bash
streamlit run app.py
```

5. **Access the Tool**
```
Open your browser and navigate to: http://localhost:8501
```

---

## 🎮 Usage Guide

### Step 1: Upload Files
- **Base File**: Your reference dataset (Excel format)
- **Check File**: The dataset you want to compare against base

### Step 2: Configure Columns
- Select **Identifier Column**: Unique ID for each entry
- Select **Text Column**: The content to analyze
- (Optional) Select **Additional Columns**: Metadata to preserve

### Step 3: Set Parameters
- **Top K Matches**: Number of similar entries to return per query (1-10)

### Step 4: Run Analysis
- Click **🚀 Run Similarity Search**
- Watch the unified progress bar track all phases:
  - Phase 1: Preprocessing & Embedding Generation
  - Phase 2: FAISS Similarity Search
  - Phase 3: LLM Relationship Analysis

### Step 5: Explore Results
- **Summary View**: High-level statistics and metrics
- **Visualization**: 3D embedding space with interactive exploration
- **Results Table**: Detailed matches with highlighted differences
- **Download**: Export as Excel (with formatting) or JSON

---

## 📊 Sample Results

### LLM Relationship Classifications

| Relationship | Score Range | Description | Example |
|-------------|-------------|-------------|---------|
| 🟢 **Exact Match** | 1.0 | Identical strings | "The quick brown fox" ↔ "The quick brown fox" |
| 🟢 **Equivalent** | 0.95-1.0 | Same meaning, different words | "automobile" ↔ "car" |
| 🟡 **Related** | 0.50-0.94 | Partial overlap or related concepts | "sedan" ↔ "vehicle" |
| 🔴 **Contradictory** | 0.00-0.20 | Opposite meanings | "hot" ↔ "cold" |

---

## 🛠️ Technical Stack

### Core Technologies
- **Framework**: Streamlit 1.28+
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: BAAI/bge-large-en-v1.5 (1024 dimensions)
- **LLM**: Groq GPT-OSS-20B via Groq API
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: Pandas, NumPy, SciKit-Learn

### Key Libraries
```
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
groq>=0.4.0
plotly>=5.14.0
openpyxl>=3.1.0
pandas>=2.0.0
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM analysis | Required |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `CUDA_AVAILABLE` | Enable GPU acceleration | `false` |

### Configurable Parameters (`app/config.py`)

```python
# Embedding Configuration
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = 32

# Text Processing
MAX_TOKENS_FOR_TRUNCATION = 512

# LLM Configuration
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TEMPERATURE = 0.0
LLM_BATCH_TOKEN_LIMIT = 100000

# Session Management
SESSION_TIMEOUT_HOURS = 24
MAX_CONCURRENT_USERS = 100
```

---

## 📁 Project Structure

```
sentence-similarity-tool/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
│
├── app/
│   ├── config.py              # Configuration & environment setup
│   ├── core.py                # FAISS indexing & similarity search
│   ├── llm_service.py         # Groq LLM integration & batching
│   ├── pipeline.py            # Orchestration of analysis workflow
│   ├── preprocess.py          # Text cleaning & normalization
│   ├── postprocess.py         # Results formatting & export
│   ├── progress_manager.py    # Unified progress tracking
│   └── utils.py               # Helper functions & utilities
│
├── app/prompts/
│   └── system_prompt.txt      # LLM prompt template
│
└── static/
    └── css/
        └── custom.css         # UI styling & themes
```

---

## 🔬 Advanced Features

### Hierarchy-Based Tie Breaking
When multiple entries have identical similarity scores, the tool uses hierarchical identifiers (e.g., "1.2.3") to select the most structurally relevant match.

### Smart Caching Strategy
- **User-Specific Cache**: Each session maintains isolated embeddings and indices
- **Hash-Based Validation**: Automatic cache invalidation on file changes
- **LLM Response Cache**: Avoid redundant API calls for identical pairs

### Token Optimization
- **Batch Aggregation**: Groups LLM calls to maximize tokens per request
- **Dynamic Batching**: Respects token limits while minimizing API calls
- **Usage Tracking**: Real-time display of prompt/completion token consumption

---

## 🎨 Screenshots

### Main Interface
![Main Interface](docs\images\Main_Interface.png)
*Upload files, select columns, and configure analysis parameters*

### Processing Pipeline
![Processing](https://via.placeholder.com/800x400?text=Unified+Progress+Tracking)
*Real-time progress across preprocessing, FAISS search, and LLM analysis*

### Results Dashboard
![Results](https://via.placeholder.com/800x400?text=Interactive+Results+%26+Visualizations)
*Comprehensive summary with FAISS and LLM insights*

### Results Table
![Table](https://via.placeholder.com/800x400?text=Detailed+Match+Results)
*Color-coded relationships with highlighted text differences*

---

## 🧪 Testing

Run tests with:
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=app tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **BAAI** for the BGE-large-en-v1.5 embedding model
- **Groq** for ultra-fast LLM inference
- **Facebook Research** for FAISS vector search library
- **Streamlit** for the amazing web framework
- **HuggingFace** for model hosting and deployment

---

## 📧 Contact

**Vignesh Manivasakam**

- GitHub: [@Vignesh-Manivasakam](https://github.com/Vignesh-Manivasakam)
- Project Link: [https://github.com/Vignesh-Manivasakam/sentence-similarity-tool](https://github.com/Vignesh-Manivasakam/sentence-similarity-tool)
- Live Demo: [https://huggingface.co/spaces/Vignesh0503/sentence-similarity-tool](https://huggingface.co/spaces/Vignesh0503/sentence-similarity-tool)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ by Vignesh Manivasakam

</div>