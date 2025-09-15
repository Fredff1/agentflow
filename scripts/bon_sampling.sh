export CUDA_VISIBLE_DEVICES=1

python /root/workspace/agent-rm/Agent-Verifier/src/bon_sampling.py \
    --config /root/workspace/agent-rm/Agent-Verifier/config/sample_0915.yaml \
    --input  /root/workspace/agent-rm/datasets/aime2025/aime2025-full.jsonl \
    --output /root/workspace/agent-rm/datasets/aime2025/0915/Qwen-2.5-7B-Instruct_bon16.jsonl \
    --batch-size 32 \
    --num-samples 16 \