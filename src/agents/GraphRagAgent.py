from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action

class GraphRagAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, backend, llm):
        super().__init__()
        self.backend = backend
        self.backend.load_model(llm)
    
    @property
    def actions(self):
        return ["graphrag"]
    
    @register_action
    def graphrag(self, prompt):
        res = self.backend._call(prompt=prompt)
        return {"result": res}

if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    # from src.backend.kernelModels.InterLM2 import internlm220b  # 导入llm模型
    # llm = internlm220b
    # llm.load_model()

    from langchain_openai import ChatOpenAI
    import os
    os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
    os.environ["OPENAI_API_KEY"] = 'sk-Gah5iODlG7OAojOK569f1103E7F9428c96E9Dc9e5954064a'
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

    from src.backend.kernelModels.GraphRAG import graphragllm  # 导入后端模型
    graphragagent = GraphRagAgent(graphragllm, llm)  # 实例化Agent 
    
    graphragagent._show()  # 展示Agent的属性
    print(graphragagent.actions)  # 展示Agent已经实现的动作
    graphragagent.execute("graphrag", prompt={"question": "我的症状为咳嗽，吐出脓血，是什么问题？"})  # 执行动作
    print(graphragagent.context.content)  # 获得动作的结果
        