from src.config.base import BaseVLMConfig

class QilinMedVL(BaseVLMConfig):
    def __init__(self):
        self.model_path = "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/Qilin-Med-VL-Chat-model"
        self.image_aspect_ratio = "pad"
        self.conv_mode = "llava_v1"
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.bit = "Full"  # "Full \ 8bit \ 4bit"
        self.device = "cuda"
        self.device_map = "auto"

qinlinconf = QilinMedVL()