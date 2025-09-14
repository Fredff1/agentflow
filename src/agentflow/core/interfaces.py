# src/core/interfaces.py
from typing import List, Tuple, Dict, Any, Protocol, Sequence, Union
from abc import ABC, abstractmethod
from .types import Query, Rollout, Verdict

class LLMBackend(ABC):
    
    @abstractmethod
    def generate(self, prompts: List, extra: List[Dict], **kwargs) -> Tuple[List[str],List[Dict]]: ...

class SupportChatTemplate(Protocol):
    def apply_chat_template(self, messages: List[Dict[str,str]], tokenize=False, 
                            add_generation_prompt=True, **additional_params) -> Union[str,Any]:...

class CanGenerate(Protocol):
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:...

class CanRMScores(Protocol):
    def score(self, sequences: Sequence[str], extra: List[Dict] = None, **kwargs) -> List[float]: ...

class CanChoiceProbs(Protocol):
    def choice_probs(self, prefixes: Sequence[str], choices: Sequence[Sequence[str]]) -> List[List[float]]: ...

class CanContLogprobs(Protocol):
    def continuation_logprobs(self, prefixes: Sequence[str], continuations: Sequence[str]) -> List[float]: ...