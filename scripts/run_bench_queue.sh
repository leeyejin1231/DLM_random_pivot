#!/bin/bash
# Usage: run_bench_queue.sh <gpu> <job1> [<job2> ...]   job = task:method[:shard]
cd /home/ssgyejin/contents/DLM_random_pivot
export HF_HUB_OFFLINE=1
PY=/home/ssgyejin/miniconda3/envs/diffuguard/bin/python
MODEL=/scratch/ssgyejin/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
D=/scratch/ssgyejin/datasets
gpu=$1; shift
for job in "$@"; do
  IFS=: read task method shard <<< "$job"
  case $task in
    math500) data=$D/HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow; gen=256;;
    mbpp)    data=$D/google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow; gen=256;;
    ifeval)  data=$D/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84/if_eval-train.arrow; gen=512;;
  esac
  case $method in
    vanilla) cfg=vanilla_variant.json;;
    hier)    cfg=hierarchy_r03_variant.json;;
    ours)    cfg=confpivot_v2_t075_p32_s2.json;;
    *)       cfg=confpivot_v2_${method}.json;;
  esac
  out=results/${task}_g${gen}_b32_${method}; extra=""
  if [ -n "$shard" ]; then out=${out}_shard${shard/\//-}; extra="--shard $shard"; fi
  echo "=== $(date) START $job -> $out" >> logs/bench_queue_gpu${gpu}.log
  CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_bench.py --task $task --model $MODEL --dataset $data \
    --output $out --gen-length $gen --block-length 32 --steps $gen $extra --variant-config $cfg \
    > logs/bench_${task}_${method}${shard:+_shard${shard/\//-}}.out 2>&1
  echo "=== $(date) END $job exit=$?" >> logs/bench_queue_gpu${gpu}.log
done
