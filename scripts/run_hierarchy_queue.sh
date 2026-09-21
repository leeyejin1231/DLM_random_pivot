#!/bin/bash
# Wait for the vanilla/random-pivot HumanEval run (PID $1) to exit, then run the Hierarchy-dLLM baseline on GPU 1.
WAIT_PID=$1
OUT=/home/yejin/contents/DLM_EF_random_pivot/results/humaneval_164_hierarchy_g512_b32_seed42
mkdir -p "$OUT"
echo "$(date) waiting for PID $WAIT_PID" >> "$OUT/queue.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "$(date) PID $WAIT_PID exited; launching hierarchy run" >> "$OUT/queue.log"
cd /home/yejin/contents/DLM_EF_random_pivot
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_MODULES_CACHE=/home/yejin/.cache/huggingface/modules \
/home/yejin/anaconda3/bin/python -u eval_humaneval.py \
  --model /mnt/shared/huggingface-cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 \
  --dataset /mnt/shared/huggingface-cache/datasets/openai___openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow \
  --gen-length 512 --block-length 32 --steps 512 \
  --variant-config /home/yejin/contents/DLM_EF_random_pivot/hierarchy_variants.json \
  --output "$OUT" > "$OUT/run.log" 2>&1
echo "$(date) hierarchy run exited with code $?" >> "$OUT/queue.log"
