 FROM python:3.10-slim

 # Set working directory
 WORKDIR /app

 # Install system dependencies (for faiss and other packages)
 RUN apt-get update && apt-get install -y \
     build-essential \
     gcc \
     && rm -rf /var/lib/apt/lists/*

 # Copy requirements first (optimizes Docker caching)
 COPY requirements.txt .

 # Install Python dependencies
 RUN pip install --no-cache-dir -r requirements.txt

 # Copy the rest of your app code
 COPY . .

 # Create necessary directories (for cache, output, prompts, logs)
 RUN mkdir -p data/cache data/output app/prompts logs static/css

 # Set environment variables for Streamlit and HF
 ENV STREAMLIT_SERVER_PORT=7860
 ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
 ENV HF_HOME=/tmp/.cache/huggingface
 ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
 ENV PYTHONUNBUFFERED=1

 # Expose the port HF Spaces expects
 EXPOSE 7860

 # Health check for HF monitoring
 HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

 # Run the Streamlit app
 ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]