from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action

class GraphRagAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, backend, llm):
        super().__init__()
        self.backend = backend
        self.backend.load_model(llm)
    
    @property
    def actions(self):
        return ["graphrag"]
    
    @register_action
    def graphrag(self, prompt):
        res = self.backend._call(prompt=prompt)
        return {"result": res}

        