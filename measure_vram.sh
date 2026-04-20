#!/bin/bash
set -euo pipefail

LOG=/workspace/vram_results.txt
: > "$LOG"

gb() { python3 -c "print(f'{$1/1024:.2f}')"; }

measure() {
    local label="$1"
    local service="$2"
    local container="$3"
    local wait_sec="${4:-300}"
    local extra_compose="${5:-}"

    echo ""
    echo "=========================================="
    echo "[$label] Starting..."
    echo "=========================================="

    baseline=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    echo "Baseline GPU0: ${baseline} MiB"

    docker compose -f /workspace/docker-compose.yml -f /workspace/docker-compose.gpu.yml \
        $extra_compose up -d $service

    echo "Waiting for healthy (max ${wait_sec}s)..."
    elapsed=0
    status=""
    while [ $elapsed -lt $wait_sec ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
        if [ "$status" = "healthy" ]; then
            echo "Healthy after ${elapsed}s"
            break
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        printf "."
    done
    echo ""

    if [ "$status" != "healthy" ]; then
        echo "WARNING: not healthy after ${wait_sec}s (status=$status), measuring anyway"
    fi

    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    vram=$((used - baseline))
    echo "VRAM [$label]: ${vram} MiB (~$(gb $vram) GB)"
    echo "RESULT:${label}:${vram}" | tee -a "$LOG"

    docker compose -f /workspace/docker-compose.yml -f /workspace/docker-compose.gpu.yml \
        $extra_compose stop $service
    docker compose -f /workspace/docker-compose.yml -f /workspace/docker-compose.gpu.yml \
        $extra_compose rm -f $service

    echo "Waiting 20s for VRAM release..."
    sleep 20

    released=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    echo "GPU0 after release: ${released} MiB"
}

# --- small models first ---
measure "MOSS-Audio-4B"                 "moss-4b"                  "arena-moss-4b"                  300
measure "MOSS-Audio-8B"                 "moss-8b"                  "arena-moss-8b"                  300
measure "MOSS-Audio-8B-Thinking"        "moss-8b-thinking"         "arena-moss-8b-thinking"         300
measure "Gemma-4-E4B"                   "gemma4-e4b"               "arena-gemma4-e4b"               300
measure "Qwen2-Audio"                   "qwen2-audio"              "arena-qwen2-audio"              300
measure "Audio Flamingo Next"           "audio-flamingo"           "arena-audio-flamingo"           300
measure "Audio Flamingo Next Captioner" "audio-flamingo-captioner" "arena-audio-flamingo-captioner" 300
measure "Audio Flamingo Next Think"     "audio-flamingo-think"     "arena-audio-flamingo-think"     300
measure "SALMONN-13B"                   "salmonn-13b"              "arena-salmonn-13b"              300 "--profile salmonn"

# --- step-audio: vllm is GPU container, r1 is wrapper ---
echo ""
echo "=========================================="
echo "[Step-Audio-R1.1] Starting vllm + r1..."
echo "=========================================="
baseline=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
echo "Baseline GPU0: ${baseline} MiB"
docker compose -f /workspace/docker-compose.yml -f /workspace/docker-compose.gpu.yml up -d step-audio-vllm step-audio-r1
echo "Waiting for step-audio-r1 healthy (max 800s)..."
elapsed=0; status=""
while [ $elapsed -lt 800 ]; do
    status=$(docker inspect --format='{{.State.Health.Status}}' arena-step-audio-r1 2>/dev/null || echo "unknown")
    if [ "$status" = "healthy" ]; then echo "Healthy after ${elapsed}s"; break; fi
    sleep 10; elapsed=$((elapsed+10)); printf "."
done
echo ""
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
vram=$((used - baseline))
echo "VRAM [Step-Audio-R1.1]: ${vram} MiB (~$(gb $vram) GB)"
echo "RESULT:Step-Audio-R1.1:${vram}" | tee -a "$LOG"
docker compose -f /workspace/docker-compose.yml -f /workspace/docker-compose.gpu.yml stop step-audio-r1 step-audio-vllm
docker compose -f /workspace/docker-compose.yml -f /workspace/docker-compose.gpu.yml rm -f step-audio-r1 step-audio-vllm
sleep 30

# --- large qwen3 models ---
measure "Qwen3-Omni"           "qwen3-omni"           "arena-qwen3-omni"           600
measure "Qwen3-Omni-Captioner" "qwen3-omni-captioner" "arena-qwen3-omni-captioner" 600
measure "Qwen3-Omni-Thinking"  "qwen3-omni-thinking"  "arena-qwen3-omni-thinking"  600

echo ""
echo "=========================================="
echo "ALL DONE. Results:"
cat "$LOG"
echo "=========================================="
