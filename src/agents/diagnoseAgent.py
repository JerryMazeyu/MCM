import re
from src.agents.baseAgent import BaseAgent
from src.utils.io import ShowConfigMixin
from src.agents.context import register_action
from src.utils.utils import log
from typing import Union, Optional
from warnings import warn
from src.utils.translator import translator_en_to_zh

class DiagnoseAgent(BaseAgent, ShowConfigMixin):
    def __init__(self, llmagent:BaseAgent, vlmagent = Optional[BaseAgent], ragagent = Optional[BaseAgent]):
        super().__init__()
        self.HISTORY = []  # Unified management of history
        self.ISDIAGNOSING = False  # If is answering medical problem
        self.llmagent = llmagent
        self.vlmagent = vlmagent
        self.ragagent = ragagent
        try:  # Loading Agents
            self.llmagent._show()
            log("The LLM model was successfully imported.")
        except:
            raise ImportError("Failed to load LLM model, please check the loading code and model path.")
        try:
            self.vlmagent._show()
            log("The VLM model was successfully imported.")
        except:
            warn("Failed to load VLM model, please check the loading code and model path.")
        try:
            self.ragagent._show()
            log("The RAG model was successfully imported.")
        except:
            warn("Failed to load RAG model, please check the loading code and model path.")

    
    @property
    def actions(self):
        return ["white_chat", "ichat", "img_chat"]

    @register_action
    def white_chat(self, prompt, with_mem=True):
        """Common chat use llm agent.

        Args:
            prompt (str): Prompt to llm.
            history (list, optional): History of the dialogue. Defaults to [].
            with_mem (bool, optional): If record this chat to the history. Defaults to True.

        Returns:
            context: Dict.
        """
        result = self.llmagent.execute("chat", prompt=prompt, history=self.HISTORY, with_mem=False)
        if with_mem:
            self.HISTORY.append((prompt, result["result"]))
        return result
    
    @register_action
    def ichat(self, prompt):
        """Intelligence chat.

        Args:
            prompt (str): Prompt to llm

        Returns:
            context: Context dict.
        """
        if self.ragagent:
            prompt_history={"question": prompt, "history":self.HISTORY}
            rag_result = self.ragagent.execute("graphrag", prompt=prompt_history)
            # print(rag_result)
        log(rag_result["result"]['answer'], level="TOUSER")
        rag_result_str = self._prettify_rag_result(rag_result["result"])
        log(f"文档索引内容：{rag_result_str}")
        self.HISTORY.append((prompt, rag_result["result"]))
        self.ans = rag_result["result"]['answer']
        self.execute("_check_answer")
        return rag_result
    
    @register_action
    def img_chat(self, inp, img):
        """Image chat.

        Args:
            inp (List(str)): [ Modality of image, Organ of image in chinese] 
            imagepath (str): Path of image. 

        Returns:
            context: Context dict.
        """
        if self.vlmagent:
            vlm_result = self.vlmagent.execute("img_chat", inp=inp, img=img)
            # vlm_result_zh = translator_en_to_zh.translate(vlm_result["result"])
            res = vlm_result["result"]
            log(f"你的器官是否健康：{res}")
        return vlm_result
    
    @register_action
    def _leadin(self, verbose=False):
        leadin_prompt = "请你扮演一个医学专家，你除了具有丰富西医知识外，你还具有中医知识，我可能会对你进行问诊、医学咨询以及医学常识问答等。"
        out = self.llmagent.execute("chat", prompt=leadin_prompt, history=self.HISTORY, with_mem=True)['history']
        self.HISTORY.append(out[0])
        if verbose:
            log(leadin_prompt, level="TOLLM")
            log(out[0][0], level="TOUSER")
        return {}

    @register_action
    def _thirdparty_chat(self, prompt):
        """Third party chat use llm agent, usually act as third person.

        Args:
            prompt (str): Prompt to llm.
            history (list, optional): History of the dialogue. Defaults to [].
            with_mem (bool, optional): If record this chat to the history. Defaults to True.

        Returns:
            context: Dict.
        """
        result = self.llmagent.execute("chat", prompt=prompt, history=[], with_mem=False)
        return result

    @register_action
    def _check_answer(self):
        if self.HISTORY != []:
            # answer = self.HISTORY[-1][1]
            if self._check1(self.ans):
                prompt = f"请你判断用户的下面这个问题是否和医学有关? 请只回答Y或N。用户的问题是：{self.HISTORY[-1][0]}"
                result = self.execute("_thirdparty_chat", prompt=prompt)["result"]
                self.ISDIAGNOSING = self._check(result)
                if not self.ISDIAGNOSING:
                    log("检测到您的问题与医学无关，您本次的对话记录将不会保留，您可以继续您的问诊问题。")
                    self.HISTORY.pop()
        return {}
    
    def _prettify_rag_result(self, rag_res):
        res = f"""
        \t\t\t 根据之前的对话，我们认为您主要的提问可以被结构化为{rag_res['reference']['structured_list']};
        \t\t\t 根据知识图谱查询结果，主要的可能疾病为(已出现症状数 / 总症状数): {rag_res['reference']['structured_data'][1]};
        \t\t\t 知识图谱的查询过程为: {rag_res['reference']['structured_data'][0]};
        \t\t\t 从已有资料中，查询到的结果为: {rag_res['reference']['unstructured_data']};
        """
        return res
    
    def _prettify_history(self, history):
        prettified_history = ""
        for q, a in history:
            prettified_history += f"问：{q} 答：{a}"
        return prettified_history

    def _check1(self, text):
        # print('text',text)
        patterns =[
        "抱歉",
        "无法",
        "无法回答",
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return False
        return True
    
    def _check(self, text):
        patterns =[
        r"(?i)\byes\b",  # 匹配 "yes"，忽略大小写
        r"(?i)\by\b",    # 匹配 "Y"，忽略大小写
        "是的",
        "没错",
        "(?=.*有关)(?!.*无关)"  # 正向先行断言匹配“有关”，负向先行断言确保没有“无关”
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    @register_action
    def _clear_history(self):
        self.HISTORY = []
        return {}
    


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    from src.agents.mockAgent import MockAgent
    mock = MockAgent()
    dagent = DiagnoseAgent(mock, mock, mock)


    res = dagent.execute("ichat", prompt="你好！")
    # res = dagent.execute("diagnose")
    # print(dagent.HISTORY)
    print(res)
    