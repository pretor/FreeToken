#!/usr/bin/env bash
set -euo pipefail

FREETOKEN_ROOT="/home/pretor/freetoken-tp4"
FREETOKEN_VENV="/home/pretor/.freetoken/venv"
CUDA_HOME_DIR="/home/pretor/cuda"
KERNEL_CACHE="${HOME}/.cache/flashinfer"

export FREETOKEN_DISK_READ_WORKERS=8

FT_GIFS="${FT_GIFS:-0,1,2,3}"
FT_PORT="${FT_PORT:-8000}"
FT_MIN_FREE_GIB="${FT_MIN_FREE_GIB:-0}"
FT_MEMORY_RATIO="${FT_MEMORY_RATIO:-0.95}"
FT_MODEL_PATH="${FT_MODEL_PATH:-/models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4-FTW}"
FT_SERVED_NAME="${FT_SERVED_NAME:-Qwen3.8-Flash-Next-NVFP4-QAD}"
FT_PLE_BACKEND="${FT_PLE_BACKEND:-pinned}"
FT_MOE_CACHE_SIZE="${FT_MOE_CACHE_SIZE:-15000}"

echo "=== launching ft serve (TP4) on gpus $FT_GIFS, port $FT_PORT, memory-ratio $FT_MEMORY_RATIO ==="
echo "ft: $FT_MODEL_PATH | quant moe.nvfp4=triton | ple $FT_PLE_BACKEND | moe offload | cache-size $FT_MOE_CACHE_SIZE | tokens 200000"

cd "$FREETOKEN_ROOT"
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$FREETOKEN_VENV/bin:$CUDA_HOME_DIR/bin:$PATH"
export LD_LIBRARY_PATH="/home/pretor/.freetoken/venv/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME_DIR/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE="$KERNEL_CACHE"
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Operational NCCL & Step Timeout Configuration
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export NCCL_TIMEOUT=120
export FREETOKEN_STEP_TIMEOUT=120
export FREETOKEN_DISTRIBUTED_TIMEOUT=2592000
export FREETOKEN_DISABLE_OVERLAP_SCHEDULING=0

exec numactl --interleave=all "$FREETOKEN_VENV/bin/ft" serve \
  --model "$FT_MODEL_PATH" \
  --tp-size 4 --gpu "$FT_GIFS" \
  --moe-strategy offload --quant-backend moe.nvfp4=triton \
  --ple-backend "$FT_PLE_BACKEND" --expert-load serial \
  --embed-device cpu \
  --moe-cache-size "$FT_MOE_CACHE_SIZE" --memory-ratio "$FT_MEMORY_RATIO" \
  --moe-prefill-hit-d2d --kv-cache-dtype fp8 \
  --num-tokens 200000 --max-seq-len-override 200000 --kv-reserve-tokens 200000 \
  --max-running-requests 2 --cuda-graph-max-bs 2 --max-extend-length 4096 --mamba-host-slots 32 \
  --served-model-name "$FT_SERVED_NAME" \
  --text-model-only \
  --step-timeout 120 --distributed-timeout 2592000 \
  --host 0.0.0.0 --port "$FT_PORT"
