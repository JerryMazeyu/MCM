from src.runner.baseRunner import BaseRunner
import warnings
warnings.filterwarnings('ignore')
from src.utils.utils import log
from src.agents.diagnoseAgent import DiagnoseAgent
from src.agents.mockAgent import MockAgent
# from src.agents.LLMAgent import LLMAgent, llmagent
from prompt_toolkit import prompt as p
import datetime
from src.agents.LLMAgent import LLMAgent
from src.agents.MedDrAgent import MedDrAgent
from src.agents.GraphRagAgent import GraphRagAgent



class DiagnoseRunner(BaseRunner):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
    
    def run(self):
        log("欢迎与问诊大模型进行对话！如希望上传图像，请回复 [IMG] 进行上传；如问诊一轮后，请回复 [FRESH] 可以开启新一轮对话。")
        while True:
            self.agent.execute("_leadin")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            prefix = f"[{current_time}] [TOLLM]>>  "
            prompt = p(prefix)
            if prompt == '[FRESH]':
                self.agent.execute("_clear_history")
                log("已为您重启新一轮问诊对话，之前的记录已清除。")
                continue
            if prompt == '[IMG]':
                img = input("请输入您的医疗影像路径：")  # /home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source369.jpg
                inp = input("请输入您的医疗影像模态和器官，用空格分开，我将告知您的器官是否健康：")  # X-Ray 肺
                inp=inp.split(' ')
                self.agent.execute("img_chat", inp=inp, img=img)
            else:
                self.agent.execute("ichat", prompt=prompt)  # 我最近头痛、发热，还身体强硬，有什么病吗？

if __name__ == "__main__":
    r = DiagnoseRunner()
    r.run()
    


