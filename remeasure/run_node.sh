#!/bin/bash
# Runs on the gman H100 node: serve each model with vLLM in turn, run its stage, stop it. Keys via env only.
set -u
cd ~; mkdir -p safety/out; export HF_HOME=$HOME/hf
serve() { # name model revision extra...
  name=$1; model=$2; rev=$3; shift 3
  nohup vllm serve $model --revision $rev --served-model-name $name --port 8000 --api-key $VLLM_API_KEY --max-model-len 8192 --gpu-memory-utilization 0.90 "$@" > safety/out/serve-$name.log 2>&1 &
  echo $! > safety/out/serve.pid
  for i in $(seq 1 360); do curl -s -o /dev/null -H "Authorization: Bearer $VLLM_API_KEY" localhost:8000/v1/models && return 0; sleep 10; done
  echo "SERVE_TIMEOUT $name"; return 1
}
stop() { kill $(cat safety/out/serve.pid) 2>/dev/null; sleep 15; pkill -f "vllm serve" 2>/dev/null; sleep 5; }
P="python3 safety/pipeline.py"
serve gen Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 5a5a776300a41aaa681dd7ff0106608ef2bc90db && $P gen --endpoint http://localhost:8000 --model gen --questions safety/questions.jsonl --out safety/out/gen.jsonl --max-tokens 1024; echo GEN_DONE $?; stop
serve judgeA openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee && $P judge --endpoint http://localhost:8000 --model judgeA --questions safety/questions.jsonl --gen safety/out/gen.jsonl --out safety/out/labels-judgeA.jsonl --rater-id judgeA --reasoning-effort low --max-tokens 4096; echo JUDGEA_DONE $?; stop
serve judgeB microsoft/phi-4 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 && $P judge --endpoint http://localhost:8000 --model judgeB --questions safety/questions.jsonl --gen safety/out/gen.jsonl --out safety/out/labels-judgeB.jsonl --rater-id judgeB --max-tokens 512; echo JUDGEB_DONE $?; stop
echo ALL_DONE
