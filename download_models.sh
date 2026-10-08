#!/bin/bash
# ==============================================================================
# Robust Overnight Model Downloader for Multimodal Video RAG
# Downloads selected reasoning LLMs and vision VLMs sequentially into Ollama.
# ==============================================================================

LOG_FILE="download_models.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo " Starting Overnight Model Download Pipeline"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Target Hardware: 6GB VRAM (Optimized 3B, 7B & 8B Models)"
echo " Logs will be saved to: $LOG_FILE"
echo "=========================================================="
echo ""

# 1. Check if Ollama binary is available
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: 'ollama' command not found in PATH."
    echo "Please make sure the Ollama installation has completed."
    exit 1
fi

# 2. Check if Ollama daemon is running; if not, start it in the background
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚙️ Ollama service is not running. Starting 'ollama serve' in background..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

# Verify connection
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama daemon is active and responding on port 11434."
else
    echo "⚠️ Warning: Could not connect to Ollama daemon on localhost:11434. Will attempt pulls directly..."
fi

echo ""

# 3. List of models using official Ollama tags
# Format: Primary tag, with fallback if tag format varies
MODELS=(
    "qwen2.5:7b"            # #1 Pick Reasoning: JSON, Math, Logic (fits in 6GB)
    "llama3.1:8b"           # Repo Baseline: General Text Reasoning
    "qwen2.5:3b"            # Ultra-fast lightweight reasoning (>50 tok/s)
    "llama3.2:3b"           # Lightweight edge conversational model
    "qwen2.5vl:7b"          # Top Pick Vision: Dynamic resolution & charts
    "minicpm-v"             # Strong Vision: High OCRBench & MathVista scores
)

TOTAL=${#MODELS[@]}
CURRENT=1
SUCCESS_COUNT=0
FAILED_MODELS=()

for MODEL in "${MODELS[@]}"; do
    echo "----------------------------------------------------------"
    echo "[$CURRENT/$TOTAL] Pulling model: $MODEL"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------------------------"

    if ollama pull "$MODEL"; then
        echo "✅ Successfully downloaded: $MODEL"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "⚠️ Failed to pull '$MODEL'. Trying fallback tag..."
        # Fallback for vision models if tag naming differs
        FALLBACK=""
        if [ "$MODEL" = "qwen2.5vl:7b" ]; then
            FALLBACK="qwen2-vl:7b"
        elif [ "$MODEL" = "minicpm-v" ]; then
            FALLBACK="minicpm-v:latest"
        fi

        if [ -n "$FALLBACK" ] && ollama pull "$FALLBACK"; then
            echo "✅ Successfully downloaded fallback: $FALLBACK"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ Could not download $MODEL. Continuing with next model..."
            FAILED_MODELS+=("$MODEL")
        fi
    fi

    echo ""
    CURRENT=$((CURRENT + 1))
done

echo "=========================================================="
echo " DOWNLOAD RUN SUMMARY"
echo " Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total models attempted: $TOTAL"
echo " Successfully downloaded: $SUCCESS_COUNT"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo " ⚠️ Failed models: ${FAILED_MODELS[*]}"
fi
echo "=========================================================="
echo ""
echo "Installed models currently in Ollama:"
ollama list || true
echo ""
echo "✨ Script completed. Log saved in $LOG_FILE."

