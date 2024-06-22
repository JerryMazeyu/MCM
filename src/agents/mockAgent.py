from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.utils.utils import log
from src.agents.context import register_action


class MockAgent(BaseAgent, ShowConfigMixin):
    def __init__(self):
        super().__init__()
        log("Loading mockagent.")

    
    @property
    def actions(self):
        return ["chat", "rag"]
    
    @register_action
    def chat(self, prompt, history=[], with_mem=True):
        return {"result": prompt, "history": [("这是一个模拟问题。", "这是一个模拟回复。")]}
    
    @register_action
    def rag(self, **kwargs):
        return {"result": "无法回答你的问题。", "reference":{"structed_input": {"症状": "头晕"}, "structed_data": (["A -> B - C - D -> E", "F - G - H"], {"高血压": "3/5", "糖尿病": "3/6"}), "unstructed_data": ["这是一个模拟参考。", "这是一个模拟参考。" ]}}
    
