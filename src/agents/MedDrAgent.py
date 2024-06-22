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

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    from src.backend.kernelModels.MedDr import meddr  # 导入后端模型
    vlmagent = MedDrAgent(meddr)  # 实例化Agent
    vlmagent._show()  # 展示Agent的属性
    print(vlmagent.actions)  # 展示Agent已经实现的动作
    # vlmagent.execute("img_chat", inp=['MRI', 'appendix'], img="/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/synpic26386.jpg")  # 执行动作
    # print(vlmagent.context.content)  # 获得动作的结果

    # inp=input("请输入您的医疗影像模态和器官，用空格分开，我将告知您的器官是否健康：")
    # # MRI 阑尾
    # inp=inp.split(' ')
    inp=['X-Ray','肺']
    # inp[1]=translator_zh_to_en.translate(inp[1])
    vlmagent.execute("img_chat", inp=inp, img="/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/LLaVA-Med/data/images/source369.jpg")  # 执行动作

    output=translator_en_to_zh.translate(vlmagent.context.content['result'])
    print(output)

    
   