import os
import torch
import requests
from PIL import Image
import argparse
import sys
sys.path.append('/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/Qilin-Med-VL')
import faulthandler
faulthandler.enable()

from io import BytesIO
from libs.llava.utils import disable_torch_init
from libs.llava.model import LlavaLlamaForCausalLM
from libs.llava.conversation import conv_templates, SeparatorStyle
from libs.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN
from libs.llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from transformers import TextStreamer, BitsAndBytesConfig, AutoTokenizer

class ImageAgent:
    def __init__(self, model_path, image_aspect_ratio, device='cuda', conv_mode='llava_v1', temperature=0.2, max_new_tokens=512, load_8bit=False, load_4bit=False):
        self.device = device
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.image_aspect_ratio = image_aspect_ratio

        # 禁用 PyTorch 的初始化，以减少启动时间
        disable_torch_init()

        # 加载模型
        self.tokenizer, self.model = self.load_model(model_path, load_8bit=False, load_4bit=False, device=None, device_map="auto")

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

    def load_model(self, model_path, load_8bit, load_4bit, device, device_map="auto"):
        kwargs = {"device_map": device_map}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **kwargs
        )
        return tokenizer, model

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
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

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
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs
        return outputs

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/Qilin-Med-VL-Chat-model")
    parser.add_argument("--image-file", type=str, default="/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/Qilin-Med-VL/playground/figures/PMC8253873_Fig6_46.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='llava_v1')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    agent = ImageAgent(
        model_path=args.model_path,
        device=args.device,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        load_8bit=args.load_8bit,
        load_4bit=True,
        image_aspect_ratio=args.image_aspect_ratio
    )

    inp = "这张图片展示了一种什么类型的医学检查?"
    outputs = agent.process_input(inp, args.image_file)
