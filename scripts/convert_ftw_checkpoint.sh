#!/usr/bin/env bash
set -euo pipefail

FREETOKEN_ROOT="/home/pretor/freetoken-tp4"
FREETOKEN_VENV="/home/pretor/.freetoken/venv"
CUDA_HOME_DIR="/home/pretor/cuda"
KERNEL_CACHE="${HOME}/.cache/flashinfer"

cd "$FREETOKEN_ROOT"
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$FREETOKEN_VENV/bin:$CUDA_HOME_DIR/bin:$PATH"
export LD_LIBRARY_PATH="/home/pretor/.freetoken/venv/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME_DIR/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export FLASHINFER_WORKSPACE_BASE="$KERNEL_CACHE"
export FLASHINFER_DISABLE_VERSION_CHECK=1

exec "$FREETOKEN_VENV/bin/ft" checkpoint   --model "/models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4"   --out "/models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4-FTW"   --dtype bfloat16   --quant-backend moe.nvfp4=triton   --gpu 0
