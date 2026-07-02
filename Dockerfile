FROM python:3.12-slim

# Install system dependencies (needed for FAISS)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy dependencies first for layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face cache directory to a local path within the app workspace
ENV HF_HOME=$HOME/app/.hf_cache

# Pre-download SentenceTransformer model weights so they are baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY --chown=user . $HOME/app

# Expose port (7860 is standard for HF Spaces, but respects PORT env var)
EXPOSE 7860

# Run with uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --log-level info"]
