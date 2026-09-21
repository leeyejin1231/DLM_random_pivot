#!/bin/bash
# Usage: run_gsm8k_he.sh <gpu> <cfgname> [gsm8k|humaneval ...]
cd /home/ssgyejin/contents/DLM_random_pivot
export HF_HUB_OFFLINE=1
PY=/home/ssgyejin/miniconda3/envs/diffuguard/bin/python
MODEL=/scratch/ssgyejin/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
gpu=$1; cfg=$2; shift 2
for task in "$@"; do
  echo "=== $(date) START $task:$cfg" >> logs/bench_queue_gpu${gpu}.log
  if [ $task = gsm8k ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_gsm8k.py --model $MODEL \
      --dataset /scratch/ssgyejin/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow \
      --output results/gsm8k_full_g128_b32_$cfg --num-samples 1319 --gen-length 128 --block-length 32 \
      --variant-config confpivot_v2_$cfg.json > logs/gsm8k_full_$cfg.out 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_humaneval.py --model $MODEL \
      --dataset /scratch/ssgyejin/datasets/openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow \
      --output results/humaneval_g256_b32_$cfg --gen-length 256 --block-length 32 \
      --variant-config confpivot_v2_$cfg.json > logs/humaneval_g256_$cfg.out 2>&1
  fi
  echo "=== $(date) END $task:$cfg exit=$?" >> logs/bench_queue_gpu${gpu}.log
done
