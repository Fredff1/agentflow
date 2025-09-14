export CUDA_VISIBLE_DEVICES=1


python /root/workspace/agent-rm/Agent-Verifier/src/score_scalar.py \
    --config /root/workspace/agent-rm/Agent-Verifier/config/scalar_rm.yaml \
    --input  /root/workspace/agent-rm/datasets/aime2025/0914/qwen2.5_7b_bon4.jsonl \
    --output /root/workspace/agent-rm/datasets/aime2025/0914/qwen2.5_7b_bon4_score_scalar.jsonl \
    --batch-size 4 \
    --use-chat-template \