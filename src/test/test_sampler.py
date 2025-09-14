import sys
import os
import logging

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from agentflow.inference.samplers.bon import BONSampler
from agentflow.backend.openai import OpenaiBackend

def test():
    cfg = {
        "backend": {
            "model_path": "deepseek-ai/DeepSeek-V3", 
            "sampling": {"temperature": 0.6, "max_tokens": 64},
            "openai": {
                "api_key": "sk-bafbshbgvwptczgoaodrjrrggwkyxluxhmjqtphyobripjaw",
                "url": "https://api.siliconflow.cn/v1",
                "max_concurrency": 2,
                "max_retries": 1,
            },
        }
    }
    backend = OpenaiBackend(cfg,logging.getLogger(__name__))
    sampler = BONSampler(
        backend,
        4,
    )
    results = sampler.sample(
        ["Hello","How are you"]
    )
    print(results)
    
if __name__ == "__main__":
    test()