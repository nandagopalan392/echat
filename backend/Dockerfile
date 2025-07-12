FROM python:3.10-slim as builder

# Install build dependencies and Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Set pip configurations
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy and install requirements first
COPY requirements.txt .

# Install dependencies with retry mechanism
RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    for i in 1 2 3; do pip install -r requirements.txt && break || sleep 15; done && \
    # Explicitly uninstall posthog to avoid compatibility issues with Python 3.8
    pip uninstall -y posthog || true

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install curl and uvicorn
RUN apt-get update && \
    apt-get install -y curl && \
    pip install "uvicorn[standard]" && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy application code
COPY . .
RUN chmod +x init_ollama.sh

# Create directories with proper permissions
RUN mkdir -p /app/data/chroma_db /app/data/db && \
    chmod -R 777 /app/data

# Expose port
EXPOSE 8000  
# Run the application
CMD ["./init_ollama.sh"]
