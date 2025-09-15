export CUDA_VISIBLE_DEVICES=1

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_agent.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/aime2025/0915/Qwen-2.5-7B-Instruct_bon16.jsonl \
  --output /root/workspace/agent-rm/datasets/aime2025/0915/Qwen-2.5-7B-Instruct_bon16_tool_score1.jsonl \
  --record-batch-size 1 \
  --join-template "Question: {prompt}\n Assistant Answer: {response}" \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
