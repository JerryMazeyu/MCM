from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action

class RagAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, backend, llm):
        super().__init__()
        self.backend = backend
        self.backend.load_model(llm)
    
    @property
    def actions(self):
        return ["rag"]
    
    @register_action
    def rag(self, question, history=[]):
        res = self.backend._call(question, history=[])
        return {"result": res}

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    from src.backend.kernelModels.InterLM2 import internlm220b  # 导入llm模型
    llm = internlm220b
    llm.load_model()

    # from langchain_openai import ChatOpenAI
    # import os
    # os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
    # os.environ["OPENAI_API_KEY"] = 'sk-Gah5iODlG7OAojOK569f1103E7F9428c96E9Dc9e5954064a'
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

    from src.backend.kernelModels.RAG import ragllm  # 导入后端模型
    ragagent = RagAgent(ragllm, llm)  # 实例化Agent 
    
    ragagent._show()  # 展示Agent的属性
    print(ragagent.actions)  # 展示Agent已经实现的动作
    ragagent.execute("rag", question="卵巢早衰怎么办？", history=[])  # 执行动作
    print(ragagent.context.content)  # 获得动作的结果

# Interlm2
# {'result': {'context': [Document(page_content='\n{"问": "卵巢早衰应该咋治恢复快，最近经常出现经量减少，经期缩短，月经周期增长，经检查是卵巢早衰，请问卵巢早衰能彻底恢复吗", "答": "卵巢早衰能彻底恢复。卵巢早衰能彻底恢复的，患者饮食方面要留意营养平衡，除了蛋白质足量摄取外，脂肪及糖类应足量，同时特别注意维生素E、D及矿物质如铁钙的消化，其中适当消化维生素E可以彻底清除自由基，稳定皮肤弹性，延后性腺膨胀的进程，起些抗衰老的作用，并可调整免疫功能，每日150—300毫克即可。要适当增强活动，活动有助于增进新陈代谢及血液循环，减缓器官衰老。活动应当量力而行持之以恒，循序渐进，如跑步、散步、广播操、太极拳均是较适宜的活动。"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 1537, 'relevance_score': 0.6189717054367065}),
#  Document(page_content='{"问": "背痛的放射治疗有些什么？", "答": "单纯体外放射治疗"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 15374, 'relevance_score': 0.4013370871543884})], 
# 'question': '卵巢早衰怎么办？', 
# 'answer': '卵巢早衰可以通过饮食和活动等方法来恢复。患者饮食方面要留意营养平衡，除了蛋白质足量摄取外，脂肪及糖类应足量，同时特别注意维生素E、D及矿物质如铁钙的消化，
# 其中适当消化维生素E可以彻底清除自由基，稳定皮肤弹性，延后性腺膨胀的进程，起些抗衰老的作用，并可调整免疫功能，每日150—300毫克即可。要适当增强活动，活动有助于增进新陈代谢及血液循环，减缓器官衰老。
# 活动应当量力而行持之以恒，循序渐进，如跑步、散步、广播操、太极拳均是较适宜的活动。

# gpt3.5
# {'result': {'context': [Document(page_content='\n{"问": "卵巢早衰应该咋治恢复快，最近经常出现经量减少，经期缩短，月经周期增长，经检查是卵巢早衰，请问卵巢早衰能彻底恢复吗", "答": "卵巢早衰能彻底恢复。卵巢早衰能彻底恢复的，患者饮食方面要留意营养平衡，除了蛋白质足量摄取外，脂肪及糖类应足量，同时特别注意维生素E、D及矿物质如铁钙的消化，其中适当消化维生素E可以彻底清除自由基，稳定皮肤弹性，延后性腺膨胀的进程，起些抗衰老的作用，并可调整免疫功能，每日150—300毫克即可。要适当增强活动，活动有助于增进新陈代谢及血液循环，减缓器官衰老。活动应当量力而行持之以恒，循序渐进，如跑步、散步、广播操、太极拳均是较适宜的活动。"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 1537, 'relevance_score': 0.6189717054367065}), 
#                         Document(page_content='{"问": "背痛的放射治疗有些什么？", "答": "单纯体外放射治疗"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 15374, 'relevance_score': 0.4013370871543884})], 
#                         'question': '卵巢早衰怎么办？', 
#                         'answer': '卵巢早衰是一种涉及生殖系统的临床问题，其治疗方案需要综合考虑患者的具体情况。一般来说，治疗卵巢早衰的方法包括以下几个方面：\n\n
#                         1. 药物治疗：这可能包括激素替代疗法，例如补充雌激素或黄体酮来调整患者的生理周期，促进卵巢功能的恢复。此外，也可能需要补充其他相关的激素或药物，例如甲状腺素等，以维持内分泌系统的平衡。\n\n
#                         2. 营养调理：通过调整饮食结构，保证患者摄取足够的营养物质，特别是蛋白质、脂肪、糖类以及维生素和矿物质等，以促进身体功能的恢复和调节内分泌系统的平衡。\n\n
#                         3. 生活方式调整：包括适当的运动和休息，保持良好的生活习惯，避免过度疲劳和精神压力，有助于调节身体的代谢和免疫功能，对卵巢功能的恢复也有积极作用。\n\n
#                         4. 心理支持：考虑到卵巢早衰可能对患者的心理和情绪产生影响，提供心理咨询和支持也是治疗的一部分，有助于患者积极面对疾病和调整心态。\n\n
#                         总体来说，治疗卵巢早衰需要综合考虑患者的身体状况、生活方式和心理健康等因素，因此建议患者在接受治疗时咨询专业医生，制定个性化的治疗方案，并且定期复查和评估疗效，以便及时调整治疗措施。'}}


        