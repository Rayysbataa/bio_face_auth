FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and face recognition
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    libgl1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with longer timeout
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Download InsightFace models during build
RUN python -c "import insightface; \
    from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
    app.prepare(ctx_id=-1, det_size=(640, 640))" || true

# Copy application code
COPY app/ ./app/
# Copy scripts if they exist
# COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/embeddings /app/data/uploads /app/data/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV VECTOR_DB_PATH=/app/data/embeddings
ENV API_HOST=0.0.0.0
ENV API_PORT=8101

# Expose port
EXPOSE 8101

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8101/ || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8101"]