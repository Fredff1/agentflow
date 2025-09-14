import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)  

import pytest

from agentflow.backend.openai import OpenaiBackend  
import logging




def test_openai_backend_smoke():
    cfg = {
        "backend": {
            "model_path": "deepseek-ai/DeepSeek-V3", 
            "sampling": {"temperature": 0.0, "max_tokens": 64},
            "openai": {
                "api_key": "sk-bafbshbgvwptczgoaodrjrrggwkyxluxhmjqtphyobripjaw",
                "url": "https://api.siliconflow.cn/v1",
                "max_concurrency": 2,
                "max_retries": 1,
            },
        }
    }
    backend = OpenaiBackend(cfg,logging.getLogger(__name__))

    texts, metas = backend.generate(["hi"])
    print(texts[0])
    assert len(texts) == 1
    out = texts[0]
    assert out is not None
    
if __name__ == "__main__":
    test_openai_backend_smoke()