export CUDA_VISIBLE_DEVICES=1,2


python /root/workspace/agent-rm/Agent-Verifier/src/score_scalar.py \
    --config /root/workspace/agent-rm/Agent-Verifier/config/scalar_rm.yaml \
    --input  /root/workspace/agent-rm/datasets/aime2025/bon/qwen3_4b_aime2025-bon128.jsonl \
    --output /root/workspace/agent-rm/datasets/aime2025/0915-analyze/qwen3_4b_aime2025-bon128_score1_scalar.jsonl \
    --batch-size 4 \
    --use-chat-template \