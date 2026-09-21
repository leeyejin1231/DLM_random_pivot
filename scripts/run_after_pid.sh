#!/bin/bash
# Usage: run_after_pid.sh <pid-to-wait-for|0> <gpu> <job> [<job> ...]
#   job = he:<cfg> (HumanEval g256) | bench:<task>:<cfg>[:shard]
cd /home/ssgyejin/contents/DLM_random_pivot
waitpid=$1; gpu=$2; shift 2
while [ "$waitpid" != 0 ] && kill -0 $waitpid 2>/dev/null; do sleep 30; done
for job in "$@"; do
  case $job in
    he:*)    "$(dirname "$0")"/run_gsm8k_he.sh $gpu ${job#he:} humaneval;;
    gsm:*)   "$(dirname "$0")"/run_gsm8k_he.sh $gpu ${job#gsm:} gsm8k;;
    bench:*) "$(dirname "$0")"/run_bench_queue.sh $gpu ${job#bench:};;
  esac
done
