export CUDA_VISIBLE_DEVICES=1

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_agent.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/aime2025/0914/qwen2.5_7b_bon4.jsonl \
  --output /root/workspace/agent-rm/datasets/aime2025/0914/qwen2.5_7b_bon4_tool_score2.jsonl \
  --record-batch-size 1 \
  --join-template "Question: {prompt}\Answer: {response}" \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
