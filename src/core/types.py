from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional

@dataclass
class Query:
    id: str
    prompt: str
    extra: Dict = field(default_factory=dict)

@dataclass
class Rollout:
    text: str
    thinking: Optional[str] = None
    meta: Dict = field(default_factory=dict)   

@dataclass
class ScoreDetail:
    verbalized: Optional[float] = None
    bool_logit: Optional[float] = None
    checklist: Optional[float] = None
    uncertainty: Optional[float] = None
    rubric: Dict = field(default_factory=dict)  
    extras: Dict = field(default_factory=dict)

@dataclass
class Verdict:
    query_id: str
    rollout_idx: int
    score: float
    detail: ScoreDetail = field(default_factory=ScoreDetail)
    chosen: bool = False

def to_dict(obj) -> Dict: return asdict(obj)
def with_update(v: Verdict, **kw) -> Verdict: return replace(v, **kw)
