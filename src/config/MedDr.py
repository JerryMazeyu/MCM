from src.config.base import BaseVLMConfig

class MedDrConfig(BaseVLMConfig):
    def __init__(self):
        self.model_path = "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedDr/model"
        self.generation_config = dict(num_beams=1,
                                      max_new_tokens=512,
                                      do_sample=False,)
        self.device = "cuda:1"
        

# meddrconf = MedDr_Config()

if __name__ == '__main__':
    meddr = MedDrConfig()
    meddr._show()