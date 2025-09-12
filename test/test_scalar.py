import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     


from src.backend.hf_scalar_rm import HFRMBackend

from src.config import load_config


def test():
    config = load_config("/root/workspace/agent-rm/Agent-Verifier/config/default.yml")
    rm_engine = HFRMBackend(config)
    messages = [
        [{"role":"user","content":"Hello,Can you tell me the value of sqrt(2)"},{"role":"assistant","content":" Yes,it is 1.414"}],
        [{"role":"user","content":"Hello,Can you tell me the value of sqrt(2)"},{"role":"assistant","content":" Yes,it is -1"}]
    ]
    prompts = rm_engine.apply_chat_template(messages)
    score = rm_engine.rm_scores(prompts)
    print(score)
    
if __name__ == "__main__":
    test()