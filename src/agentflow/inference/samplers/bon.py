# sampler.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

from agentflow.core.interfaces import CanGenerate

class BONSampler:
    """Best-of-N sampler
    """
   

    def __init__(
        self,
        backend: CanGenerate,
        num_samples: int,
        *,
        batch_size: Optional[int] = None,
        seed_base: Optional[int] = None,
        seed_key: str = "seed",
        per_sample_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.backend = backend
        self.num_samples = int(num_samples)
        assert self.num_samples >= 1, "num_samples must be >= 1"
        self.batch_size = batch_size
        self.seed_base = seed_base
        self.seed_key = seed_key
        self.per_sample_overrides = dict(per_sample_overrides or {})

    def sample(
        self,
        prompts: List,
        extra: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
        n_items = len(prompts)
        if n_items == 0:
            return [], []

        if extra is None:
            extra = [None] * n_items
        if len(extra) != n_items:
            raise ValueError(f"`extra` length {len(extra)} must match prompts length {n_items}")

        flat_prompts: List = []
        flat_extra: List[Dict[str, Any]] = []
        src_index_of_flat: List[int] = []    
        sample_id_of_flat: List[int] = []     

        for i, (p, e) in enumerate(zip(prompts, extra)):
            base_extra = dict(e or {})
            for k, v in self.per_sample_overrides.items():
                base_extra[k] = v
            for k in range(self.num_samples):
                flat_prompts.append(p)
                this_extra = dict(base_extra)
                this_extra["_source_index"] = i
                this_extra["_sample_id"] = k
                if self.seed_base is not None and (self.seed_key not in this_extra):
                    this_extra[self.seed_key] = int(self.seed_base + i * self.num_samples + k)
                flat_extra.append(this_extra)
                src_index_of_flat.append(i)
                sample_id_of_flat.append(k)

        flat_texts: List[str] = [None] * len(flat_prompts)  
        flat_metas: List[Dict[str, Any]] = [None] * len(flat_prompts)  

        def _run_chunk(start: int, end: int):
            chunk_prompts = flat_prompts[start:end]
            chunk_extra = flat_extra[start:end]
            texts, metas = self.backend.generate(chunk_prompts, extra=chunk_extra, **kwargs)
            for idx, (t, m) in enumerate(zip(texts, metas)):
                flat_texts[start + idx] = t
                flat_metas[start + idx] = m

        if self.batch_size and self.batch_size > 0:
            for s in range(0, len(flat_prompts), self.batch_size):
                _run_chunk(s, min(s + self.batch_size, len(flat_prompts)))
        else:
            _run_chunk(0, len(flat_prompts))

        out_texts: List[List[str]] = [[] for _ in range(n_items)]
        out_metas: List[List[Dict[str, Any]]] = [[] for _ in range(n_items)]

        ordering = list(range(len(flat_prompts)))
        ordering.sort(key=lambda idx: (src_index_of_flat[idx], sample_id_of_flat[idx]))
        for idx in ordering:
            i = src_index_of_flat[idx]
            out_texts[i].append(flat_texts[idx])
            out_metas[i].append(flat_metas[idx])

        return out_texts, out_metas

    def sample_one(self, prompt, extra: Optional[Dict[str, Any]] = None, **kwargs):
        texts, metas = self.sample([prompt], extra=[extra or {}], **kwargs)
        return texts[0], metas[0]
