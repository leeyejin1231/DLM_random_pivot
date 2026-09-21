#!/bin/bash
# Usage: run_dream.sh <gpu> <task:method[:shard]>
# Runs one Dream-v0-Instruct-7B evaluation with the same protocol as the LLaDA runs.
cd /home/ssgyejin/contents/DLM_random_pivot
export HF_HUB_OFFLINE=1
PY=/home/ssgyejin/miniconda3/envs/diffuguard/bin/python
MODEL=/scratch/ssgyejin/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae
D=/scratch/ssgyejin/datasets
gpu=$1; IFS=: read task method shard <<< "$2"
case $task in
  gsm8k)     data=$D/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow; gen=128; script=eval_gsm8k.py;;
  humaneval) data=$D/openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow; gen=256; script=eval_humaneval.py;;
  math500)   data=$D/HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow; gen=256; script=eval_bench.py;;
  mbpp)      data=$D/google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow; gen=256; script=eval_bench.py;;
  ifeval)    data=$D/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84/if_eval-train.arrow; gen=512; script=eval_bench.py;;
esac
case $method in
  vanilla) cfg=configs/vanilla_variant.json;;
  hier)    cfg=configs/hierarchy_r03_variant.json;;
  ours)    cfg=configs/confpivot_v2_t075_p32_s2.json;;
  wino)    cfg=configs/wino_${task}.json;;
esac
out=results/dream_${task}_g${gen}_b32_${method}; extra=""
[ -n "$shard" ] && { out=${out}_shard${shard/\//-}; extra="--shard $shard"; }
[ "$script" = eval_bench.py ] && extra="$extra --task $task"
[ "$script" = eval_gsm8k.py ] && extra="$extra --num-samples 1319"
log=logs/dream_${task}_${method}${shard:+_shard${shard/\//-}}.out
echo "=== $(date) START $2 -> $out" >> logs/dream_pool_gpu${gpu}.log
CUDA_VISIBLE_DEVICES=$gpu $PY -u $script --model-family dream --model $MODEL --dataset $data \
  --output $out --gen-length $gen --block-length 32 --steps $gen $extra --variant-config $cfg > $log 2>&1
echo "=== $(date) END $2 exit=$?" >> logs/dream_pool_gpu${gpu}.log
