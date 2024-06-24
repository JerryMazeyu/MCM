from PIL import Image
import torch
from transformers import LlamaTokenizer
from libs.Med.model.internvl_chat import InternVLChatModel
from libs.Med.dataset.transforms import build_transform
import os
from safetensors.torch import safe_open
from src.config.MedDr import MedDrConfig
from src.backend.kernelModels.baseModel import BaseLLM
from src.utils.io import LoaderMixin
import faulthandler
faulthandler.enable()
from pydantic import Extra
import re
from src.utils.translator import translator_en_to_zh, translator_zh_to_en


class MedDr(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)

    # 加载导入模型参数的字典
    def load_safetensors_model(self, model_dir):
        state_dict = {}
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.safetensors'):
                file_path = os.path.join(model_dir, file_name)
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        return state_dict
    
    # 找出模型A中有但模型B中没有的参数，并进行名称转换
    def convert_param_name(self, param_name):
        # 将 base_layer 形式转换为直接形式
        param_name = re.sub(r'\.base_layer', '', param_name)
        return param_name
    
    def load_model(self):
        missing_params_in_b = {}
        state_dict_a = self.load_safetensors_model(self.model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.model = InternVLChatModel.from_pretrained(self.model_path, torch_dtype=torch.bfloat16)
        # , device_map= 'auto'
        # .to(self.device).eval()
        params_b = dict(self.model.named_parameters())

        # 找出模型A中有但模型B中没有的参数，并进行名称转换
        for name, param in state_dict_a.items():
            converted_name = self.convert_param_name(name)
            if converted_name not in params_b:
                missing_params_in_b[name] = param

        # 更新模型B的参数
        for name, param in state_dict_a.items():
            converted_name = self.convert_param_name(name)
            if converted_name in params_b:
                params_b[converted_name].data.copy_(param.data)

        # 再次调用 eval() 确保模型在推理模式
        self.model.to(self.device).eval()

        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        pad2square = self.model.config.pad2square
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        self.image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)

    def process_input(self, inp, image_file):
        image = Image.open(image_file).convert('RGB')
        image = self.image_processor(image).unsqueeze(0).to(self.device).to(torch.bfloat16)

        modality=inp[0]
        organ=inp[1]
        # self.instruction_prompt=  f"You are a helpful medical assistant. You are required to give the diagnosis of a {organ} {modality} image.  What diseases are included in the picture?"
        # self.instruction_prompt= f"You are a helpful medical assistant.  Your task is disease diagnosis.  You are given a {modality} image. Answer the question： Is the {organ} healthy?"
        self.instruction_prompt= f"You are a helpful medical assistant.  You are given a {modality} image. Answer the question and give your reason.  Is the {organ} healthy?"
        
        
        outputs = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=image,
            question=self.instruction_prompt,
            generation_config=self.generation_config,
            print_out=False,
            device=self.device
        )
        return outputs
    
    def _call(self, inp, image_file):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            # inp=inp.split(' ')
            inp[1]=translator_zh_to_en.translate(inp[1])

            outputs = self.process_input(inp, image_file)
            output_zh=translator_en_to_zh.translate(outputs)
            
            return output_zh

    @property
    def _llm_type(self):
        return "MedDr"
    
# meddr = MedDr(meddrconf)
    
if __name__=="__main__":
# instruction_prompt="You are a helpful medical assistant. Your task is disease diagnosis. \
#             You are given a MRI image.\
#             Is the appendix healthy?"
    meddr = MedDr(meddrconf)

    llm = MedDr(meddrconf)
    llm.load_model()

    # image_file_all = [ "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source155.jpg",
    #                   "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source132.jpg"]
    # inp = ['X-Ray', 'heart']
    
    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source53.jpg",
    #      "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source71.jpg",
    #     "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source42.jpg"]
    # inp = ['MRI', 'brain']

    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source0.jpg"]
    # inp = ['MRI', 'liver']

    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source229.jpg"]
    # inp = ['CT', 'liver']

    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/synpic23555.jpg"]
    # inp = ['CT', 'renal']


    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source106.jpg"]
    # inp = ['MRI', 'lung']

    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/synpic18436.jpg",
    #                    "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/synpic26386.jpg",
    #                     "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/synpic29662.jpg"]
    # inp = ['MRI', 'appendix']

    image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source140.jpg",
                      "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source155.jpg",]
    #                   "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source149.jpg",
    #                   "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source160.jpg",
    #               "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source189.jpg",
    #               "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source173.jpg",
    #             "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source121.jpg",
    #               "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/synpic42807.jpg",
    #               
    inp = ['X-Ray', '肺']

    # image_file_all = ["/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/ISIC_0024981.jpg"]
    # inp=["dermatology", "skin"]


    for image_file in image_file_all:
        response = llm._call(inp, image_file)
        print(response)

        import json
        answers_file = "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedDr/ans_MedDr.jsonl"
        with open(answers_file, 'a', encoding='utf-8') as ans_file:
            answer = {
                "image": image_file,
                "answer": response,
                "prompt": llm.instruction_prompt
            }
            ans_file.write(json.dumps(answer, ensure_ascii=False) + "\n")
