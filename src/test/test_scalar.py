import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     


from agentflow.backend.hf_scalar_rm import HFRMBackend

from agentflow.config import load_config


def test():
    config = load_config("/root/workspace/agent-rm/Agent-Verifier/config/default.yml")
    rm_engine = HFRMBackend(config)
    messages = [
        [{"role":"user","content":"Hello,Can you tell me the value of sqrt(2)"},{"role":"assistant","content":" Yes,it is 1.414"}],
        [{"role":"user","content":"Hello,Can you tell me the value of sqrt(2)"},{"role":"assistant","content":" Yes,it is -1"}]
    ]
    prompts = rm_engine.apply_chat_template(messages)
    score = rm_engine.score(prompts)
    print(score)
    
if __name__ == "__main__":
    test()