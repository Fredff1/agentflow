export CUDA_VISIBLE_DEVICES=1,2

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_agent.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/aime2025/bon/qwen3_4b_aime2025-bon128.jsonl \
  --output /root/workspace/agent-rm/datasets/aime2025/0915/qwen3_4b_aime2025-bon128_tool_score2_by_qwen2.5-7b.jsonl \
  --record-batch-size 1 \
  --join-template "Question: {prompt}\n Assistant Answer: {response}" \
  --include_full_meta \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
