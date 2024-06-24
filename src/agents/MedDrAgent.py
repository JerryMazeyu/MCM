from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action
from src.utils.translator import translator_en_to_zh, translator_zh_to_en

class MedDrAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.backend.load_model()
    
    @property
    def actions(self):
        return ["img_chat"]
    
    @register_action
    def img_chat(self, inp, img):
        '''
        Args:
            inp (List(str)): [modality, organ]
            img (str): Image path
        '''
        res = self.backend._call(inp, img)
        return {"result": res}


    
   