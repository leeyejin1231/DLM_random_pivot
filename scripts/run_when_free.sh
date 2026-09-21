#!/bin/bash
# Usage: run_when_free.sh <gpu> <cfg> [<cfg> ...]  -- waits until the GPU is idle, then runs HumanEval g256 for each cfg
cd /home/ssgyejin/contents/DLM_random_pivot
gpu=$1; shift
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)" -gt 2000 ]; do sleep 60; done
for cfg in "$@"; do ./run_gsm8k_he.sh $gpu $cfg humaneval; done
