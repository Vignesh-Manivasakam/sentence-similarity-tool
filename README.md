---
title: Sentence Similarity Analysis Tool
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# Sentence Similarity Analysis Tool

A Streamlit app for analyzing similarity between sentences in Excel files using embeddings (E5-large-v2) and LLM analysis (Phi-3-mini).

## Demo
[Link to HF Space once deployed]

## Local Setup
1. Clone repo: `git clone https://github.com/Vignesh-Manivasakam/sentence-similarity-tool.git`
2. Install deps: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Features
- Upload two Excel files
- Select columns for IDs/text
- FAISS-based similarity search
- LLM-powered relationship analysis
- Visualizations and downloads

Built with Streamlit, SentenceTransformers, Transformers, and Plotly.