#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

# Deploy Qwen/Qwen3-0.6B on a single GPU
# python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy Qwen/Qwen3-4B on a single GPU
# python -m minisgl --model "Qwen/Qwen3-4B"

# Deploy Qwen/Qwen3-4B with TP
# python -m minisgl --model "Qwen/Qwen3-4B" --tp 2

# AWQ TP 1
# python -m minisgl \
# --model "Qwen/Qwen3-4B-AWQ" \
# --max-seq-len-override 8196 \
# --memory-ratio 0.70 \
# --port 1919 --dtype float16

# Bench Config
python -m minisgl \
--model "Qwen/Qwen3-4B" \
--max-seq-len-override 4096