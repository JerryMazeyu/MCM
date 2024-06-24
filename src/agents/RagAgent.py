from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action

class RagAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, backend, llm):
        super().__init__()
        self.backend = backend
        self.backend.load_model(llm)
    
    @property
    def actions(self):
        return ["rag"]
    
    @register_action
    def rag(self, question, history=[]):
        res = self.backend._call(question, history=[])
        return {"result": res}

