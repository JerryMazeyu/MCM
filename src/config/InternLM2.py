from src.config.base import BaseLLMConfig

class InternLM2_20bConfig(BaseLLMConfig):
    def __init__(self):
        self.version = 'final'
        self.weights_dict = {
            'default': "/mnt/share_data/internlm2-20b",
            'chat': "/mnt/share_data/internlm2-chat-20b",
            'v1': "/home/ubuntu/PuyuanChallenge/Dist/train/sft_2000iter_merged",
            'v2': "/home/ubuntu/PuyuanChallenge/Dist/train/sft_3000iter_merged",
            'final': "/home/ubuntu/PuyuanChallenge/Dist/src/assets/checkpoints/InternLM2_20b_chat_sft"
        }
        self.tokenizer_name = self.weights_dict[self.version]
        self.device = "cuda:0"


# class InternLM2_7bConfig(BaseLLMConfig):
#     def __init__(self):
#         self.model = 'AutoModelForCausalLM()'
#         self.tokenizer = 'AutoTokenizer()'
#         self.version = 'v1'
#         self.weights_dict = {
#             'v1': 'xxx',
#             'v2': 'xxx'
#         }
#         self.generate_conf = {"max_new_token": 256, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}


internlm220bconf = InternLM2_20bConfig()
# internlm27bconf = InternLM2_7bConfig


if __name__ == '__main__':
    interlm = InternLM2_20bConfig()
    interlm._show()