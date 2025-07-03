#!/bin/bash

# Wait for Ollama to become available and pull models before starting the app
# Updated to provide more robust model checking, pulling, and automatic monitoring

MAX_RETRIES=30
RETRY_INTERVAL=5
OLLAMA_URL="${OLLAMA_HOST:-http://ollama:11434}"
STATUS_URL="${OLLAMA_URL}/api/version"

# Define exact model names to ensure consistency
EMBEDDING_MODEL="mxbai-embed-large"
LLM_MODEL="deepseek-r1"  # Updated from deepseek-r1:latest to deepseek-r1

echo "Checking for Ollama availability at $STATUS_URL"

# Loop until Ollama is available or max retries reached
for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i of $MAX_RETRIES: Checking Ollama..."
    if curl -s --connect-timeout 5 "$STATUS_URL" > /dev/null; then
        echo "Ollama is available!"
        
        # Check if models already exist
        echo "Checking if required models are already available..."
        MODELS_JSON=$(curl -s "${OLLAMA_URL}/api/tags")
        
        # Extract model names using grep and cut
        if [ $? -eq 0 ]; then
            EXISTING_MODELS=$(echo "$MODELS_JSON" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
            echo "Found models: $EXISTING_MODELS"
            
            # Check for embedding model - exact match
            if echo "$EXISTING_MODELS" | grep -q "^$EMBEDDING_MODEL$"; then
                echo "Embedding model $EMBEDDING_MODEL already exists, skipping pull"
            else
                echo "Pulling embedding model $EMBEDDING_MODEL..."
                curl -X POST ${OLLAMA_URL}/api/pull -d "{\"name\":\"$EMBEDDING_MODEL\"}"
                echo "Embedding model pulled"
            fi
            
            # Check for LLM model - exact match
            if echo "$EXISTING_MODELS" | grep -q "^$LLM_MODEL$"; then
                echo "LLM model $LLM_MODEL already exists, skipping pull"
            else
                echo "Pulling LLM model $LLM_MODEL..."
                curl -X POST ${OLLAMA_URL}/api/pull -d "{\"name\":\"$LLM_MODEL\"}"
                echo "LLM model pulled"
            fi
            
            echo "Model checks completed"
        else
            # Fallback if we can't check models
            echo "Couldn't check existing models, pulling them anyway"
            echo "Pulling embedding model $EMBEDDING_MODEL..."
            curl -X POST ${OLLAMA_URL}/api/pull -d "{\"name\":\"$EMBEDDING_MODEL\"}"
            
            echo "Pulling LLM model $LLM_MODEL..."
            curl -X POST ${OLLAMA_URL}/api/pull -d "{\"name\":\"$LLM_MODEL\"}"
        fi
        
        # Export models to environment for use in Python code
        export OLLAMA_EMBEDDING_MODEL=$EMBEDDING_MODEL
        export OLLAMA_LLM_MODEL=$LLM_MODEL
        export MODELS_CHECKED=true
        
        break
    fi
    
    if [ $i -eq $MAX_RETRIES ]; then
        echo "Ollama is not available after $MAX_RETRIES attempts. Starting server without models."
    else
        echo "Ollama not ready yet, waiting $RETRY_INTERVAL seconds..."
        sleep $RETRY_INTERVAL
    fi
done

# Install dependencies
pip install -r requirements.txt

# Start monitoring as a background process if enabled
# Check if monitoring is explicitly disabled
if [[ "${DISABLE_MONITORING:-false}" != "true" ]]; then
    echo "Initializing monitoring system..."
    
    # Create prometheus metrics directory if it doesn't exist
    mkdir -p /tmp/prometheus_metrics
    
    # Set metrics directory permissions
    chmod 777 /tmp/prometheus_metrics
    
    # Start prometheus node exporter in the background
    if command -v node_exporter >/dev/null 2>&1; then
        echo "Starting node_exporter..."
        node_exporter --web.listen-address=:9100 &
        echo "Node exporter started on port 9100"
    else
        echo "Node exporter not found, skipping"
    fi
    
    # Export metrics path for the application
    export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_metrics
    export METRICS_PORT=8001
    
    echo "Monitoring initialized - metrics will be available on port 8001"
else
    echo "Monitoring disabled by environment variable"
fi

# Test chatbot initialization before starting the server
echo "Testing chatbot initialization..."
python -c "from rag import get_chatpdf_instance; instance = get_chatpdf_instance(); print('ChatPDF instance created successfully')"
if [ $? -ne 0 ]; then
    echo "WARNING: Chatbot initialization test failed, but continuing anyway..."
fi

# Start the server
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}

# Start with proper settings for metrics collection
exec uvicorn main:app --host $HOST --port $PORT --reload --log-level info
