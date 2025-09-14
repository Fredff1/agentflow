export CUDA_VISIBLE_DEVICES=1

python /root/workspace/agent-rm/Agent-Verifier/src/bon_sampling.py \
    --config /root/workspace/agent-rm/Agent-Verifier/config/test.yaml \
    --input  /root/workspace/agent-rm/datasets/aime2025/aime2025-full.jsonl \
    --output /root/workspace/agent-rm/datasets/aime2025/0914/qwen3_4b_bon4.jsonl \
    --batch-size 32 \
    --num-samples 4 \