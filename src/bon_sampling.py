# bon_sampling.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentflow.inference.samplers.bon import BONSampler
from agentflow.backend.vllm import VllmBackend
from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.math.answer_parser import evaluate_samples  

DEFAULT_TEMPLATE = """You are an excellent math expert.
Answer the given question:

Question: {question}
"""

def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_jsonl_stream(path: str, *, max_records: Optional[int] = None):
    """逐行读取 JSONL，返回 (idx, obj)"""
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

def build_prompt(block: Dict[str, Any], template: str) -> str:
    try:
        return template.format_map(block)
    except KeyError:
        q = block.get("question", "")
        return template.format_map({"question": q})

def init_sampler(config: Dict[str, Any], num_samples: int, enable_thinking: bool = False) -> BONSampler:
    backend = VllmBackend(config)
    backend.set_chat_template_defaults(enable_thinking=bool(enable_thinking))
    return BONSampler(backend=backend, num_samples=int(num_samples))

def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # 常见字段回退
        for k in ("text", "output", "answer", "content", "message"):
            if k in x:
                v = x[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def flush_batch(
    sampler: BONSampler,
    blocks: List[Dict[str, Any]],
    prompts_msgs: List[List[Dict[str, str]]],
    prompts_raw: List[str],
    output_path: str,
    *,
    first_write_mode: str,
    answer_field: str,
) -> str:
    """采样并写入；返回下一次写入应使用的 mode（'a'）"""
    if not blocks:
        return first_write_mode

    samples_2d, _ = sampler.sample(prompts_msgs)  # BON 返回二维 samples

    # 逐条写回 + 判分
    for block, prompt_text, samples in zip(blocks, prompts_raw, samples_2d):
        out_record = dict(block)
        out_record["prompt"] = prompt_text
        out_record["samples"] = samples
        out_record["type"] = "math"

        # 判分（若存在标准答案）
        gt = block.get(answer_field, None)
        if gt is not None:
            evaluations = evaluate_samples([_to_text(s) for s in samples], gt)
            for idx, eva in enumerate(evaluations):
                eva["idx"] = idx
            out_record["evaluations"] = evaluations
            num_correct = sum(1 for e in evaluations if e.get("correct"))
            out_record["num_correct"] = int(num_correct)
            out_record["any_correct"] = bool(num_correct > 0)
            out_record["accuracy"] = (float(num_correct) / max(1, len(samples)))
        else:
            out_record["evaluations"] = []
            out_record["num_correct"] = 0
            out_record["any_correct"] = False
            out_record["accuracy"] = 0.0

        JsonUtil.write_jsonlines(output_path, out_record, mode=first_write_mode)
        if first_write_mode == "w":
            first_write_mode = "a"
    return "a"

def sample_streaming(
    config_path: str,
    input_path: str,
    output_path: str,
    *,
    batch_size: int,
    num_samples: int,
    append: bool,
    template: str,
    enable_thinking: bool,
    max_records: Optional[int],
    answer_field: str,
):
    # 初始化
    config = load_config(config_path)
    sampler = init_sampler(config, num_samples=num_samples, enable_thinking=enable_thinking)
    ensure_parent_dir(output_path)

    # 首写模式：append=True 则直接 'a'；否则先 'w' 清空文件
    write_mode = "a" if append else "w"
    if not append:
        JsonUtil.write_jsonlines(output_path, [], mode="w")  # 清空

    blocks: List[Dict[str, Any]] = []
    prompts_msgs: List[List[Dict[str, str]]] = []
    prompts_raw: List[str] = []
    total = 0

    for _, block in read_jsonl_stream(input_path, max_records=max_records):
        prompt_text = build_prompt(block, template)
        messages = [{"role": "user", "content": prompt_text}]

        blocks.append(block)
        prompts_msgs.append(messages)
        prompts_raw.append(prompt_text)

        if len(blocks) >= batch_size:
            write_mode = flush_batch(
                sampler,
                blocks,
                prompts_msgs,
                prompts_raw,
                output_path,
                first_write_mode=write_mode,
                answer_field=answer_field,
            )
            total += len(blocks)
            blocks.clear()
            prompts_msgs.clear()
            prompts_raw.clear()

    # 处理尾批
    if blocks:
        write_mode = flush_batch(
            sampler,
            blocks,
            prompts_msgs,
            prompts_raw,
            output_path,
            first_write_mode=write_mode,
            answer_field=answer_field,
        )
        total += len(blocks)

    print(f"[DONE] Wrote {total} records to {output_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BON sampler (streaming JSONL writer with math parsing & checking)")
    p.add_argument("--config", required=True, type=str, help="Path to backend config (YAML/TOML)")
    p.add_argument("--input", required=True, type=str, help="Input JSONL")
    p.add_argument("--output", required=True, type=str, help="Output JSONL")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size of prompts")
    p.add_argument("--num-samples", type=int, default=4, help="Number of samples per prompt")
    p.add_argument("--append", action="store_true", help="Append to output JSONL instead of overwrite")
    p.add_argument("--max-records", type=int, default=None, help="Only process first N records")
    p.add_argument("--template-file", type=str, default=None, help="Optional template file path")
    p.add_argument("--enable-thinking", action="store_true", help="Enable thinking in chat template (default off)")
    p.add_argument("--answer-field", type=str, default="answer", help="Key of ground-truth answer in input JSONL record")
    return p.parse_args()

def main():
    args = parse_args()

    # 模板
    template = DEFAULT_TEMPLATE
    if args.template_file:
        with open(args.template_file, "r", encoding="utf-8") as f:
            template = f.read()

    sample_streaming(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        batch_size=max(1, int(args.batch_size)),
        num_samples=max(1, int(args.num_samples)),
        append=bool(args.append),
        template=template,
        enable_thinking=bool(args.enable_thinking),
        max_records=args.max_records,
        answer_field=args.answer_field,
    )

if __name__ == "__main__":
    main()
