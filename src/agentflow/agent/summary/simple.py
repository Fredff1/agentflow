from __future__ import annotations
from typing import Protocol, List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

from agentflow.common.messages import Message, trans_messages_to_text
from agentflow.core.interfaces import CanGenerate
from .interface import SummarizerInterface, SummaryItem
from agentflow.utils.chat_template import is_chat_messages

def default_prompt_template(content: str) -> str:
    return (
        "你是一个精炼的对话总结助手。请在不丢失关键事实的前提下进行高信噪比总结。\n"
        "【输出格式】\n"
        "- 摘要：...\n- 结论/决定：...\n- 行动项：...\n- 未决问题：...\n\n"
        "=== 原文 ===\n"
        f"{content}\n"
    )

@dataclass
class GeneratorSummarizer(SummarizerInterface):
    generator: CanGenerate
    prompt_template_fn: Callable[[str],str] = default_prompt_template  

    def summarize(self, item: SummaryItem, **kwargs) -> Tuple[str, Dict[str, Any]]:
        outs, metas = self.summarize_batch([item], **kwargs)
        return outs[0], metas[0]

    def summarize_batch(
        self,
        items: List[SummaryItem],
        **kwargs
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts: List[str] = []
        for it in items:
            if isinstance(it, str):
                texts.append(it)
            else:
                texts.append(trans_messages_to_text(it))

        prompts = [self.prompt_template_fn(t) for t in texts]
        messages = [[{"role":"user","content":prompt}] for prompt in prompts]
        outputs, gen_metas = self.generator.generate(messages, extra=None, **kwargs)

        metas = [{"generator_meta": m} for m in gen_metas]
        return outputs, metas