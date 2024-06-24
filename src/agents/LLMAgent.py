from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action

class LLMAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.backend.load_model()
    
    @property
    def actions(self):
        return ["chat"]
    
    @register_action
    def chat(self, prompt, history=[], with_mem=True):
        res, history_ = self.backend._call(prompt, history=history, only_res=False)
        if with_mem:
            return {"result": res, "history": history_}
        else:
            return {"result": res, "history": history}







        