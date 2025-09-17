from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Callable

from .context import AgentContext
from agentflow.core.interfaces import CanGenerate
from agentflow.common.messages import Message, trans_messages_to_standard, trans_messages_to_text
from agentflow.tools.base import ToolCallResult
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.caller import ToolCaller       
from agentflow.tools.base import ToolParser      
from agentflow.utils.chat_template import is_chat_messages   

class ToolDrivenAgent(CanGenerate):
    """An agent which can conduct multi-turn generation with tool-usage
    """
    def __init__(
        self,
        backend: CanGenerate,         
        tool_caller: ToolCaller,  
        finish_fn: Callable[[AgentContext],bool],
        error_fn: Optional[Callable[[AgentContext],bool]] = None,
        *,     
        max_rounds: int = 6,    
    ):
        """Initialize agent
        
        Args:
            backend (CanGenerate): Backend for basic generation
            tool_caller (ToolCaller): Tool caller ti conduct tool-calls
            finish_fn (Callable[[AgentContext],bool]): A function to decide when to finish multi-turn generation.It returns true when geneation is over.
            error_fn (Optional[Callable[[AgentContext],bool]], optional): A function to decide when an output is invalid and requires roll-back.It returns true when finding an invalid generation Defaults to None.
            max_rounds (int, optional): Max rounds for multi-turn generation. Defaults to 6.
        """
        self.backend = backend
        self.tool_caller = tool_caller
        self.tool_parser = tool_caller.parser
        self.max_rounds = max_rounds
        self.finish_fn = finish_fn
        self.error_fn = error_fn

    def generate(
        self, 
        prompts: List, 
        extra: List[Dict] = None, 
        **kwargs
    ) -> Tuple[List[str],List[Dict]]:
        prompt_len = len(prompts)
        
        if prompt_len < 1:
            raise ValueError("Prompts cannot be empty")
        if isinstance(prompts[0],str):
            tmp = [[{"role":"user","content":prompt}] for prompt in prompts]
            prompts = tmp
        elif is_chat_messages(prompts[0]):
            pass
        else:
            raise ValueError("Prompts must be a list of str or chat message lists")
        extra = extra or [None] * prompt_len
        msgs_per_samp: List[List[Message]] = [Message.from_dicts(prompt) for prompt in prompts]
        contexts: List[AgentContext] = [AgentContext.from_messages(msgs,ex) for msgs, ex in zip(msgs_per_samp,extra)]
        
        final_texts = ["" for _ in range(prompt_len)]
        finished = [False] * prompt_len
        metas: List[Dict] = [{"steps": [], "context":contexts[i]} for i in range(prompt_len)]
        for round in range(self.max_rounds):
            active = [i for i in range(prompt_len) if not finished[i]]
            
            if not active:
                break
            
            batch_inputs = [trans_messages_to_standard(contexts[i].all_messages()) for i in active]
            batch_extra = [contexts[i].meta for i in active]
            texts, backend_metas = self.backend.generate(batch_inputs, extra=batch_extra, **kwargs)
            
            for j, i in enumerate(active):
                contexts[i].step_index = round
                metas[i]["steps"].append({"assistant_text": texts[j], "backend_meta": backend_metas[j]})
                contexts[i].append_assistant(texts[j])
                
            calls_per_text = self.tool_parser.parse_batch(texts, backend_metas)
            should_finish_flags: List[bool] = []
            needs_tool_flags: List[bool] = []
            for j, i in enumerate(active):
                has_calls = len(calls_per_text[j]) > 0
                needs_tool_flags.append(has_calls)
                should_finish_flags.append(self.finish_fn(contexts[i]))
                if (not has_calls) and (not self.finish_fn(contexts[i])):
                    if self.error_fn and self.error_fn(contexts[i]):
                        contexts[i].pop() 
            
            for j, i in enumerate(active):
                if should_finish_flags[j]:
                    finished[i] = True

            to_call_local_indices: List[int] = [j for j in range(len(active)) if (not should_finish_flags[j]) and needs_tool_flags[j]]
            if not to_call_local_indices:
                if all(finished):
                    break
                continue

            for j in to_call_local_indices:
                i = active[j]
                contexts[i].meta.setdefault("round_counter", {})
                contexts[i].meta["round_counter"].update(contexts[i].round_counters)

            sub_texts = [texts[j] for j in to_call_local_indices]
            sub_metas = [backend_metas[j] for j in to_call_local_indices]  
            batch_results_sub: List[List[ToolCallResult]] = self.tool_caller.call_batch(sub_texts, sub_metas)

            for k, j in enumerate(to_call_local_indices):
                i = active[j]
                results = batch_results_sub[k]
                metas[i]["steps"][-1]["tool_results"] = results
                for r in results:
                    contexts[i].update_round_counter(r)
                    obs_text = self.tool_parser.make_result_str(r)
                    contexts[i].append_tool_observation(obs_text, metadata=r.meta)
                    
        for i in range(prompt_len):
            assistant_messages = contexts[i].all_round_messages()
            final_texts[i] = trans_messages_to_text(assistant_messages)

        return final_texts, metas
            