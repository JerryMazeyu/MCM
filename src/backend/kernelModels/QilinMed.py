# import sys
# sys.path.append('/home/ubuntu/PuyuanChallenge/Dist/libs')
# sys.path.append('/home/ubuntu/PuyuanChallenge/Dist/src')
from src.config.QilinMed import qinlinconf
from src.backend.kernelModels.baseModel import BaseLLM
from src.utils.io import LoaderMixin
from libs.llava.utils import disable_torch_init
from libs.llava.model import LlavaLlamaForCausalLM
from libs.llava.conversation import conv_templates, SeparatorStyle
from libs.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN
from libs.llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from transformers import TextStreamer, BitsAndBytesConfig, AutoTokenizer
import torch
import requests
from PIL import Image
from io import BytesIO
import faulthandler
faulthandler.enable()
from pydantic import Extra


class QilinMedVL(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        disable_torch_init()
    
    def load_model(self):
        kwargs = {"device_map": self.device_map}

        if self.bit == "8bit":
            kwargs['load_in_8bit'] = True
        elif self.bit == "4bit":
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )
        # self.model = self.model.eval()

        # 添加图像标记
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.vision_tower = self.model.get_vision_tower()
        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model()
        self.vision_tower.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dtype=torch.float16)
        self.image_processor = self.vision_tower.image_processor
        self.context_len = 2048

        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles
    
    def load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def process_input(self, inp, image_file):
        # 处理图像，将其转换为张量，并移动到指定的设备上。
        image = self.load_image(image_file)
        image_tensor = process_images([image], self.image_processor, self)
        # print(image_tensor.shape)#torch.Size([1, 3, 336, 336])
        if type(image_tensor) is list:
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # 添加图像标记，并将用户输入添加到对话中。生成提示。
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.roles[0], inp)
        self.conv.append_message(self.roles[1], None)
        prompt = self.conv.get_prompt()

        # 将提示转换为输入 ID，并设置停止条件。
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # input_ids = input_ids.long()  # 确保input_ids的类型为LongTensor

        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        # output_ids = self.model(image_tensor, input_ids)

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs
        return outputs

    def _call(self, inp, image_file):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            outputs = self.process_input(inp, image_file)
            return outputs

    @property
    def _llm_type(self):
        return "Qilin-Med-VL"
    
qilinmed = QilinMedVL(qinlinconf)

import os
import json
# def get_jpg_files(image_folder):
#     return [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# def extract_filename(file_path):
#     # 获取文件名（包括扩展名）
#     file_name_with_extension = os.path.basename(file_path)
#     # 去掉扩展名，获取纯文件名
#     file_name = os.path.splitext(file_name_with_extension)[0]
#     return file_name

if __name__ == '__main__':
    llm = QilinMedVL(qinlinconf)
    llm.load_model()

    # outputs = llm._call("图片中包括哪些疾病？", "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source160.jpg")
    # vqa-med-2021 synpic26386、18436|图像中出现了什么异常|急性阑尾炎 #根据您提供的信息，这张计算机断层扫描图像显示了右侧肾盂结石。肾盂结石是在肾盂内形成的结石，可能会导致尿路梗阻和尿路不适。
    # slake "xmlab160/source.jpg", “图片中包括哪些疾病？”，“答案”：“结节” ，#根据图片标题提供的信息，图片中展示了三种疾病：结节性硬化症、肺结核和结节性硬化症转变。
    # “xmlab173/source.jpg”，“问题”：“图片中包含哪些疾病？”，“答案”：“肺炎” #根据图片标题提供的信息，图片中展示了三种疾病：结节、斑块和浸润。
    # “xmlab124/source.jpg”，“问题”：“图片中包括哪些疾病？”，“答案”：“心脏肥大”，#根据您提供的信息，胸部X光片显示了一个异常的、增大的心脏轮廓，并且肺野清晰。然而，我无法直接查看图像，无法确定具体的异常。建议您咨询医疗专业人士，他们可以根据图像和其他临床信息来进行评估和诊断。
    
    image_folder = "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images"
    image_file="source189.jpg"
    image_path = os.path.join(image_folder, image_file)
    question="你是个乐于助人的医疗助理。你的任务是辅助诊断。您需要根据X-Ray图像回答以下问题。请问肺部是否健康？"
    outputs = llm._call(question, image_path)
    # 图片中包含哪些疾病？
    # 肺部健康吗？
    answers_file = "/home/ubuntu/PuyuanChallenge/Dist/src/backend/kernelModels/ans.jsonl"

    
    with open(answers_file, 'a', encoding='utf-8') as ans_file:
        answer = {
            "image": image_file,
            "answer": outputs,
            "question": question
        }
        ans_file.write(json.dumps(answer, ensure_ascii=False) + "\n")

        # for image_file in image_files:
        #     image_path = os.path.join(image_folder, image_file)
        #     question = "图片中包括哪些疾病？"
            
        #     outputs = llm._call(question, image_path)
        #     print(outputs)
            
        
    