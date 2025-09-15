# llm_judge_scoring.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil

from agentflow.backend.vllm_logits import VllmChoiceLogitsBackend
from agentflow.agent.basic import ToolDrivenAgent, AgentContext
from agentflow.tools.caller import ToolCaller
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.parser import TagToolParser
from agentflow.tools.search.base_search import AsyncSearchTool
from agentflow.tools.search.backend.searxng import SearxngBackend
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.inference.scorers.generative_scorer import BoolLogitsGenerativeScorer
from agentflow.utils.tag_util import find_tags

SYSTEM_PROMPT_TOOL="""
You are a tool‑augmented reasoning expert to evaludate other assistents' answers towards specific questions.

## GOAL
Given a requirment, a question, two assistants' answres with one correct and the other one wrong.Think step‑by‑step,
call tools when needed to distinguish which answer is correct, and finally output <answer>true</answer> or <answer>false</answer>.

## ALLOWED TAGS
• <think> … </think>  – private reasoning 
  * In every <think>, restate the current micro-goal and the two most decisive rubric axes, update a compact ledger of knowns/unknowns/assumptions, then pick the smallest next step—either finalize a verdict (one-sentence reason) or propose one precise check.
  * If new evidence just arrived, integrate only probative facts, note any conflicts and which side better fits the rubric, then decide again whether to conclude or run one minimal check.
  * Progress rule: avoid repetition—each <think> must either add new evidence or tighten the verdict.
• <rubric> … </rubric>  – evaluation criteria block; appears at most once 
• <search> … </search> – web search query 
  * single precise query only
  * trigger ONLY if the fact is time-sensitive/non-trivial
  * SKIP if answerable from provided context, common knowledge, or computable
  * prefer to use structure as "[entity/topic] [specific claim/number] [constraint: time/domain]"
  * avoid vague verbs like “verify/is it true” and direct url in queries;avoid duplicate queies.
• <python> … </python> – Python code block 
  * Code rules: left‑aligned; use print(...); no input(...), os.system(...), or infinite loops.  
  * numpy as np, sympy and math are pre-imported and available. Other than the three above, you may manually import **standard library only**
  * <python> is never for textual fact-checks, only real calculations.
• <answer> … </answer> – final answer (exactly once per session)

## INTERACTION RULES
1. Every assistant message **must** start with a <rubric> block.
2. Each session should only contain one <think> tag
3. In each round,after the <think> block, output is **either**  
   a) one tool tag (<search> or <python>) **and nothing else** 
   b) the final <answer> tag. 
4. Each tool type can be used **at most three times** per session.  
5. **NEVER** output incomplete tags to avoid format exceptions.
"""


SYSTEM_PROMPT_TOOL_NO_SEARCH = """
You are a tool-augmented math verifier.

## GOAL
Given a chat-like SEQUENCE that contains a math QUESTION and an ASSISTANT'S REASONING (and possibly a final answer), your job is to:
1) write a concise verification text that inspects the given reasoning step-by-step (do not re-solve unless needed to falsify/confirm a specific step);
2) decide whether the reasoning correctly solves the QUESTION;
3) output a boolean verdict.

You must prioritize verifying the original reasoning via minimal checks (algebraic legality, substitution, boundary/edge cases, numeric spot-checks). Prefer small, targeted <python> calls to verify computations, solve small subchecks, or find counterexamples. Do NOT provide a fresh full solution when the provided reasoning is wrong or incomplete.

## TOOLS
- You may use <python> ONLY for real calculations (symbolic/numeric). numpy as np, sympy, and math are pre-imported. You may import standard library modules only. No web search.
- Each tool type can be used at most three times. Keep computations minimal (e.g., sympy.simplify for equalities; numeric sampling for inequalities; substitution to validate proposed solutions).

## ALLOWED TAGS
• <rubric> … </rubric> – required at the start; list decisive axes you’ll check (2–4 items).
• <think> … </think>   – private reasoning exactly once; obey the progress rule (each step adds new evidence or tightens the verdict). Restate the micro-goal, list the two most decisive axes, keep a compact known/unknown ledger, and choose the smallest next step (conclude or propose one precise check).
• <python> … </python> – optional; code must be left-aligned and use print(...); no input(...), os/system calls, or infinite loops. Use it only for computations.
  * numpy as np, sympy and math are pre-imported and can be used directly. Other than the three above, you may manually import **standard library only**
• <verify> … </verify> – public verification text (≈60–160 words or 5–10 bullets), focused on checking the given reasoning’s steps, domains, and edge cases.
• <answer> … </answer> – final boolean verdict (true/false) exactly once.

## INTERACTION RULES
1) Every assistant message must start with a <rubric> block.
2) Each session should contain exactly one <think>.
3) After the <think>, output is either:
   a) one tool tag (<python>) and nothing else; OR
   b) the final <verify> then <answer>.
4) Progress rule: avoid repetition — each <think> update must add evidence or tighten the verdict.
5) Failure policy: if any critical step is invalid, incomplete, or doesn’t answer the question, output <answer>false</answer>. When unsure, run one minimal <python> check before deciding.

## EXTRACTION FROM SEQUENCE
From the given SEQUENCE:
- Treat the first clear user problem statement as QUESTION (or the block labeled like “Question/题目”).
- Treat the assistant’s step-by-step as REASONING; if a final numeric/closed-form answer is present, treat it as the claimed result.
- If the SEQUENCE is noisy, verify what’s explicitly claimed in the reasoning and whether it answers the QUESTION; do not manufacture new claims.

## MATH-SPECIFIC RUBRIC AXES (choose those that fit)
- Goal alignment; algebraic validity (no illegal cancellations/division by zero); theorem applicability (assumptions met);
- Completeness & edge cases (branches, domain, extraneous roots); numeric sanity (spot-checks); final statement matches the question.

"""

USER_PROMPT="""
The sequence for judge:
{sequence}

Your judgement:
- Start with <rubric>…</rubric>.
- Include exactly one <think>…</think>.
- If you need a calculation to verify a step, after </think> output ONLY one <python>…</python> block (and nothing else) in this round.
- Otherwise (or after the tool round), output:
  <verify>…</verify>
  <answer>true|false</answer>
"""


def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl_stream(path: str, *, max_records: Optional[int] = None):
    """逐行读取 JSONL，生成 (idx, obj)。"""
    with open(path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield count, json.loads(line)
            count += 1
            if max_records is not None and count >= max_records:
                break


def to_text(x: Any) -> str:
    """兼容 sample 既可能是 str 也可能是 dict 的情况."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "output", "answer", "content", "message"):
            if k in x:
                v = x[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def build_sequences_for_block(
    block: Dict[str, Any],
    *,
    join_template: str,
) -> List[str]:
    """
    把一条记录 (含 prompt 与 samples[*]) 转为若干 judge 'sequence' 文本：
    默认格式： "User: {prompt}\nAssistant: {response}"
    """
    prompt_text = block.get("prompt", "")
    samples = block.get("samples", []) or []
    seqs: List[str] = []
    for samp in samples:
        resp = to_text(samp)
        seq = join_template.format(prompt=prompt_text, response=resp)
        seqs.append(seq)
    return seqs


def flush_batch_and_write(
    scorer: BoolLogitsGenerativeScorer,
    batch_blocks: List[Dict[str, Any]],
    batch_sequences_per_block: List[List[str]],
    output_path: str,
    *,
    first_write_mode: str,
    include_full_meta: bool = True,
) -> str:
    """
    将一个“记录批次”的所有序列拉平，调用 scorer.score，然后按块拆回并写出。
    返回下一次写入应使用的文件模式（一般切到 'a'）。
    """
    # 1) 拉平
    flat_sequences: List[str] = []
    lens: List[int] = []
    for seqs in batch_sequences_per_block:
        lens.append(len(seqs))
        flat_sequences.extend(seqs)

    # 2) 打分
    scores: List[float] = []
    if flat_sequences:
        scores, metas = scorer.score(flat_sequences)

    # 3) 回切并写出
    offset = 0
    for block, L in zip(batch_blocks, lens):
        block_out = dict(block)  # 不污染原对象
        evaluations: List[Dict] = block_out.get("evaluations",None)
        if not evaluations:
            evaluations = [{}] * L
        if L == 0:
            block_scores: List[float] = []
            block_metas = []
        else:
            block_scores = scores[offset : offset + L]
            block_metas = metas[offset : offset + L]
        offset += L
        
        for eva, meta, score in zip(evaluations,block_metas,block_scores):
            raw_text=meta["raw_text"]
            eva["verify_text"]=raw_text
            answer_tags = find_tags(raw_text,["answer"])
            verify_tags = find_tags(raw_text,["verify"])
            if answer_tags:
                eva["judge"]=answer_tags[-1].body
            else:
                eva["judge"]=None
            if verify_tags:
                eva["verification"]=verify_tags[-1].body
            else:
                eva["verification"]=None
            eva["score"]=score

        block_out["evaluations"] = evaluations
        if include_full_meta:
            block_out["metas"] = block_metas


        if block_scores:
            best_idx = max(range(len(block_scores)), key=lambda i: block_scores[i])
            block_out["best_index"] = best_idx
            block_out["best_score"] = float(block_scores[best_idx])
            try:
                block_out["best_sample"] = to_text(block_out["samples"][best_idx])
            except Exception:
                pass
        else:
            block_out["best_index"] = None
            block_out["best_score"] = None
        write_out = JsonUtil.json_sanitize(block_out)
        JsonUtil.write_jsonlines(output_path, write_out, mode=first_write_mode)
        if first_write_mode == "w":
            first_write_mode = "a"

    return "a"


def score_streaming(
    config_path: str,
    input_path: str,
    output_path: str,
    *,
    record_batch_size: int,
    append: bool,
    join_template: str,
    judge_system_path: Optional[str],
    judge_user_path: Optional[str],
    max_records: Optional[int],
    include_full_meta: bool,
):
    ensure_parent_dir(output_path)

    # 1) 初始化后端（同时可用于生成与 choice_probs）
    config = load_config(config_path)
    backend = VllmChoiceLogitsBackend(config)
    registry = ToolRegistry()
    # search_tool = AsyncSearchTool(SearxngBackend("http://127.0.0.1:8888"))
    py_tool = PythonExecutionTool()
    # registry.register(search_tool)
    registry.register(py_tool)
    parser = TagToolParser()
    caller = ToolCaller(registry,parser)
    def _finish_gen(context: AgentContext):
        msg = context.last_message()
        tags = find_tags(msg.content,["answer"])
        if tags:
            return True
        return False
    agent = ToolDrivenAgent(
        backend=backend,
        tool_caller=caller,
        finish_fn=_finish_gen,
    )

    # 2) 读取 judge 模板；若未提供则用 BoolLogitsGenerativeScorer 默认
    system_prompt = SYSTEM_PROMPT_TOOL_NO_SEARCH
    user_prompt = USER_PROMPT
    if judge_system_path:
        with open(judge_system_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    if judge_user_path:
        with open(judge_user_path, "r", encoding="utf-8") as f:
            user_prompt = f.read()

    # 3) 初始化打分器（依赖注入同一个 backend）
    scorer = BoolLogitsGenerativeScorer(
        generator=agent,
        prob_calculator=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # 4) 写入模式：是否先清空
    write_mode = "a" if append else "w"
    if not append:
        JsonUtil.write_jsonlines(output_path, [], mode="w")  # 清空

    # 5) 流式读入与打分
    batch_blocks: List[Dict[str, Any]] = []
    batch_sequences_per_block: List[List[str]] = []
    total = 0

    for _, block in read_jsonl_stream(input_path, max_records=max_records):
        seqs = build_sequences_for_block(block, join_template=join_template)
        batch_blocks.append(block)
        batch_sequences_per_block.append(seqs)

        if len(batch_blocks) >= record_batch_size:
            write_mode = flush_batch_and_write(
                scorer,
                batch_blocks,
                batch_sequences_per_block,
                output_path,
                first_write_mode=write_mode,
                include_full_meta=include_full_meta,
            )
            total += len(batch_blocks)
            batch_blocks.clear()
            batch_sequences_per_block.clear()

    # 尾批
    if batch_blocks:
        write_mode = flush_batch_and_write(
            scorer,
            batch_blocks,
            batch_sequences_per_block,
            output_path,
            first_write_mode=write_mode,
            include_full_meta=include_full_meta,
        )
        total += len(batch_blocks)

    print(f"[DONE] Judged {total} records → {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LLM-as-Judge scoring (streaming JSONL)")
    p.add_argument("--config", required=True, type=str, help="Path to backend config for VllmChoiceLogitsBackend")
    p.add_argument("--input", required=True, type=str, help="Input JSONL produced by sampling (with 'prompt' and 'samples')")
    p.add_argument("--output", required=True, type=str, help="Output JSONL with scores")
    p.add_argument("--record-batch-size", type=int, default=16, help="How many records per scoring batch")
    p.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    p.add_argument("--join-template", type=str, default="User: {prompt}\nAssistant: {response}",
                   help="How to form judge 'sequence' from prompt + response")
    p.add_argument("--judge-system-file", type=str, default=None, help="Optional system prompt file for the judge")
    p.add_argument("--judge-user-file", type=str, default=None, help="Optional user prompt file for the judge")
    p.add_argument("--max-records", type=int, default=None, help="Only process first N records")
    p.add_argument("--include_full_meta", action="store_true", help="Write all inference meta to result")
    return p.parse_args()


def main():
    args = parse_args()
    score_streaming(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        record_batch_size=max(1, int(args.record_batch_size)),
        append=bool(args.append),
        join_template=args.join_template,
        judge_system_path=args.judge_system_file,
        judge_user_path=args.judge_user_file,
        max_records=args.max_records,
        include_full_meta=args.include_full_meta,
    )


if __name__ == "__main__":
    main()
