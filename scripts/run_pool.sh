#!/bin/bash
# Usage: run_pool.sh <gpu> <jobs_file>
# Pops jobs one at a time under flock, so each GPU always runs exactly one job.
cd /home/ssgyejin/contents/DLM_random_pivot
gpu=$1; jobs=$2; lock=$jobs.lock; cursor=$jobs.cursor
touch "$lock"; [ -f "$cursor" ] || echo 0 > "$cursor"
while true; do
  job=$(flock "$lock" bash -c "i=\$(cat '$cursor'); n=\$(wc -l < '$jobs'); if [ \$i -lt \$n ]; then echo \$((i+1)) > '$cursor'; sed -n \"\$((i+1))p\" '$jobs'; fi")
  [ -z "$job" ] && break
  scripts/run_dream.sh "$gpu" "$job"
done
echo "=== $(date) POOL DONE gpu$gpu" >> logs/dream_pool_gpu${gpu}.log
