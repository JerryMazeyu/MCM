from src.config.InternLM2 import internlm220bconf
from src.backend.kernelModels.baseModel import BaseLLM
from src.utils.io import LoaderMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

import torch
from pydantic import BaseModel


class InternLM2_20b(BaseLLM, LoaderMixin):
    class Config:
        extra = 'allow'

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        # self.version = version
        if self.weights_dict[self.version]:
            self.weights = self.weights_dict[self.version]
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.weights, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        self.model = self.model.eval()
    
    # def _call(self, prompt, history=[]):
    #     if not hasattr(self, 'model'):
    #         raise ValueError("Have not load model yet, please run llm.load_model() first!")
    #     else:
    #         if history != []:
    #             prettified_history = "历史会话记录: "
    #             for q, a in history:
    #                 prettified_history += f"问: {q} 答: {a} "
    #             prettified_history += "现在的问题: "
    #             prompt = f"{prettified_history} {prompt}"
    #         input_ids = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
    #         # generation_config = GenerationConfig(**self.generate_conf)
    #         output = self.model.generate(
    #             input_ids,
    #             **self.generate_conf,
    #             pad_token_id=self.tokenizer.eos_token_id,
    #             # streamer=streamer
    #         )
    #         response = self.tokenizer.decode(output[0], skip_special_tokens=True)
    #         history.append((prompt, response))
    #         return response, history
    
    def _call(self, prompt, history=[], stop=None, only_res=True, **kwargs):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            response, history = self.model.chat(self.tokenizer, prompt, history)
            if only_res:
                return response
            else:
                return response, history

    @property
    def _llm_type(self):
        return "InternLLM2-20b"
    
# internlm220b = InternLM2_20b(internlm220bconf)
    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    llm = internlm220b
    llm.load_model()
    res = llm._call("Hello")
    print(res)