import os
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough, RunnableLambda
import os
import re
import json
from src.backend.kernelModels.baseModel import BaseLLM
from src.utils.io import LoaderMixin
# from src.backend.kernelModels.RAG import unstructured_retriever
from pydantic import BaseModel, Extra
# from src.config.RAG import graphragconf
import inspect
from src.backend.kernelModels.InterLM2 import InternLM2_20b
from src.config.InternLM2 import internlm220bconf


class GraphRAG(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf, retriever) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        self.uns_retriever = retriever
        os.environ["NEO4J_URI"] = self.neo4j['url']
        os.environ["NEO4J_USERNAME"] = self.neo4j['username']
        os.environ["NEO4J_PASSWORD"] = self.neo4j['password']
        if self.backend == 'gpt-4o':
            os.environ["HTTP_PROXY"] = self.gpt4o['http_proxy']
            os.environ["HTTPS_PROXY"] = self.gpt4o['https_proxy']
            os.environ["OPENAI_API_KEY"] = self.gpt4o['apikey']
            self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        # self.struct_model = InternLM2_20b(internlm220bconf, version='chat')
        # self.struct_model.load_model()

    def find_and_parse_json(self, input_string:str):
        json_pattern = re.compile(r'```json(.*?)```', re.DOTALL)
        matches = json_pattern.search(input_string)
        # 初始化默认值
        disease = None
        symptom = None
        med = None
        if matches:
            json_content = matches.group(1).strip()
            try:
                data = json.loads(json_content)
                disease = data.get("疾病")
                symptom = data.get("症状") # ['恶心', '干呕', '头疼']
                med = data.get("方剂")
            except json.JSONDecodeError as e:
                print("JSON解析错误:", e)
        else:
            print("未找到JSON内容")
        return {"疾病": disease, "症状": symptom, "方剂": med}

    def structify_output(self, model, prompt, question, struct_dict_keys):
        verified_prompt =f"{prompt}, 用{','.join([x for x in struct_dict_keys])}的JSON形式回复."
        prompt = PromptTemplate.from_template(verified_prompt)
        chain=({"question": RunnablePassthrough(), "history": RunnablePassthrough()} | prompt | model | StrOutputParser())
        res=chain.invoke({"question": question})
        # print(res)
        return self.find_and_parse_json(res)

    def structured_output(self, question):
        prompt= f"你的任务是从问题中提取疾病、症状和方剂，如果问题中没有提到则为None。问题：{question}"
        # ，如果没有则为空
        # {' '.join([x[0] for x in history])}
        # model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        # model = ChatOpenAI(model="gpt-4o", temperature=0)
        struct_dict_keys = ["疾病", "症状", "方剂"]
        # res=self.structify_output(self.struct_model, prompt, question, struct_dict_keys)
        res=self.structify_output(self.model, prompt, question, struct_dict_keys)
        return res

    def structured_retriever(self, structured_list: list[str]) -> str:
        """
        Collects the neighborhood of entities (symptoms) mentioned
        in the list of symptoms.
        """
        # structured_list = {'疾病': ["少阴病"], '症状': ['发热','呕吐']}
        self.graph = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password ="neo4jneo4j")
        # result = ""
        result = []
        possible_diseases=[]
        diseases_probability={}
        if structured_list["症状"]:
            for symptom in structured_list["症状"]:
                response = self.graph.query(
                    """ 
                    CALL {
                    MATCH (node)-[r:临床表现]->(neighbor:临床表现 {name:$symptom})
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    }
                    RETURN output LIMIT 10
                    """,
                    {"symptom": symptom},
                )
                # 提取所有疾病名称
                for item in response:
                    possible_diseases.append(item['output'].split(' - ')[0])
                    result.append(item['output'])  # 三元组
                # result += "\n".join([el['output'] for el in response])

            # 计算疾病概率
            probability=[]
            for disease in possible_diseases:
                probability.append(self.disease_probability(structured_list["症状"], disease))
            
            # 概率字典
            for key, value in zip(possible_diseases, probability):
                diseases_probability[key] = value

        if structured_list["疾病"]:
            for disease in structured_list["疾病"]:
                response = self.graph.query(
                    """ 
                    CALL {
                    MATCH (node {name:$disease})-[r:治疗]->(neighbor)-[t:用法用量]->(medicine)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name + ' - ' + type(t) + ' -> ' + medicine.name AS output
                    UNION
                    MATCH (node {name:$disease})-[r:治疗]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$disease})-[r:并发症]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$disease})-[r:主治]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$disease})-[r:病因]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    }
                    RETURN output LIMIT 10
                    """,
                    {"disease": disease},
                )
                for item in response:
                    result.append(item['output'])  # 三元组

        if structured_list["方剂"]:
            for med in structured_list["方剂"]:
                response = self.graph.query(
                    """ 
                    CALL {
                    MATCH (node {name:$med})-[r:入药部位]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$med})-[r:功效]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$med})-[r:性味]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$med})-[r:成份]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$med})-[r:用法用量]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    UNION
                    MATCH (node {name:$med})-[r:注意事项]->(neighbor)
                    RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name AS output
                    }
                    RETURN output LIMIT 10
                    """,
                    {"med": med},
                )
                for item in response:
                    result.append(item['output'])  # 三元组
            # print("result", result)
                    # MATCH (node {name:$disease})-[r:治疗]->(neighbor)-[t:用法用量]->(medicine)
                    # RETURN node.name + ' - ' + type(r) + ' -> ' + neighbor.name + ' - ' + type(t) + ' -> ' + medicine.name AS output

        return result, diseases_probability

    # 计算症状占比
    def disease_probability(self, symptom, disease):
        response = self.graph.query(
                """ 
                CALL {
                MATCH (node:疾病{name:$disease})-[r:临床表现]->(neighbor)
                RETURN neighbor.name AS output
                }
                RETURN output LIMIT 10
                """,
                {"disease": disease},
            )
        all_symptom=[el['output'] for el in response]
        count = sum(1 for s in symptom if s in all_symptom)
        probability = f"{count}/{len(all_symptom)}" if all_symptom else 0
        return probability
    
    def retriever(self, question):
        # print(question) #独立问题：我感到心烦，但不再呕吐了，这是什么症状？
        structured_list = self.structured_output(question)
        structured_data = self.structured_retriever(structured_list)
        # print("structured_data: ", structured_data)
        unstructured_data = self.uns_retriever.invoke(question)
        unstructured_texts = [el.page_content for el in unstructured_data]
        # print("unstructured_texts: ", unstructured_texts)
        # 将 unstructured_data 中的 Document 对象转换为字符串
        # unstructured_texts = [doc.text for doc in unstructured_data]

        # 创建包含 structured_data 和 unstructured_texts 的字典
        reference = {
            "structured_list": structured_list,
            "structured_data": structured_data,
            "unstructured_data": unstructured_data
        }
        return reference
    
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_texts)}
            """
        return final_data

    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human in chat_history:
            buffer.append(HumanMessage(content=human))
            # buffer.append(AIMessage(content=ai))
        return buffer

    def load_chain(self, llm, verbose=False):
        # Condense a chat history and follow-up question into a standalone question
        # 提取出所有症状，
        # 给出以下对话历史和后续问题，。用中文总结成一个独立问题。用中文将后续问题改写为一个独立问题。
        _template = """给出以下对话历史和后续问题，理解对话历史和后续问题，用中文总结成一个独立问题。
        对话历史：
        {history}
        后续问题: {question}
        独立问题："""  # noqa: E501
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
            buffer = []
            for human, ai in chat_history:
                buffer.append(HumanMessage(content=human))
                # buffer.append(AIMessage(content=ai))
            return buffer

        _search_query = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
                RunnableLambda(lambda x: bool(x.get("history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),  # Condense follow-up question and chat into a standalone_question
                RunnablePassthrough.assign(
                    history=lambda x: _format_chat_history(x["history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | self.model
                # | ChatOpenAI(model="gpt-4o", temperature=0)
                | StrOutputParser(),
            ),
            # llm
            # ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            # self.struct_model
            # ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x : x["question"]),
        )

        template = """使用以下的参考上下文来回答最后的问题。如果你不知道答案，就说你不知道。
        请提供详细而清晰的回答。
        可参考的上下文：
        ···
        {reference}
        ···
        问题: {question}
        有用的回答:"""

        rag_prompt = PromptTemplate.from_template(template)
    
        # # 返回source
        # rag_chain_from_docs = (
        # RunnablePassthrough.assign(context=(lambda x: self.str_reference(x["reference"])))
        # | rag_prompt
        # | llm
        # | StrOutputParser()
        # )

        # rag_chain_with_source = RunnableParallel(
        #     {"reference": self.retriever, "question": RunnablePassthrough()}
        # ).assign(answer=rag_chain_from_docs)

        # return rag_chain_with_source 

        rag_chain_from_docs = ( rag_prompt
        | llm
        | StrOutputParser()
        )
        # RunnablePassthrough.assign(context=(lambda x: self.str_reference(x["reference"]))) # 返回的
        # |

        chain = RunnableParallel(
            { "reference":  _search_query | self.retriever ,
              "question": RunnablePassthrough(), 
            }
        ).assign(answer=rag_chain_from_docs)

        return chain

        # chain = (
        #     RunnableParallel(
        #         {
        #             "reference": _search_query | self.retriever,
        #             "question": RunnablePassthrough(),
        #         }
        #     )
        #     | rag_prompt
        #     | llm
        #     | StrOutputParser()
        # )

        # return chain

    def str_reference(self, x):
        unstructured_str = "\n\n".join(doc.page_content for doc in x["unstructured_data"])
        final_data = f"""Structured data: {x["structured_data"]}. Unstructured data: {unstructured_str}"""
        return final_data

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def load_model(self, llm):
        self.model = self.load_chain(llm)
    
    def _call(self, prompt):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            response = self.model.invoke(prompt)
            # {"question": prompt, "history": history}
            # ['result']
            return response

    @property
    def _llm_type(self):
        return "GraphRAG"


# ============================================================
# Example wirte by Jerry

# def structify_output(model, prompt, struct_dict_keys):
#     import json
#     verified_prompt = f"{prompt}, respond in JSON with {','.join([x for x in struct_dict_keys])}."
#     raw_result = model(verified_prompt)  
#     try:
#        return json.dumps(raw_result)
#     except:
#        return None
    

if __name__ == '__main__':
    graphragllm = GraphRAG(graphragconf)

    # from langchain_community.llms import FakeListLLM
    # responses = ["Hi, Jerry"]
    # llm = FakeListLLM(responses=responses)
    # res = llm.invoke("hello")
    # class Joke(BaseModel):
    #     setup: str = Field(description="The setup of the joke")
    #     punchline: str = Field(description="The punchline to the joke")
    # structured_llm = llm.with_structured_output(Joke)
    # a = structured_llm.invoke("Tell me a joke about cats")
    # print(a)

    # class AnswerWithJustification(BaseModel):
    #     '''An answer to the user question along with justification for the answer.'''
    #     answer: str
    #     justification: str

    # import re
    # import json
    # def find_and_parse_json(input_string:str):
    #     pattern = r'\{(?:[^{}]|(?R))*\}'
    #     matches = re.finditer(pattern, input_string, re.DOTALL)
    #     for match in matches:
    #         try:
    #             json_obj = json.loads(match.group(0))
    #             return json_obj
    #         except json.JSONDecodeError:
    #             continue
    #     return None

    # def structify_output(model, prompt, question, struct_dict_keys):
    #     import json
    #     verified_prompt =f"{prompt}, 用{','.join([x for x in struct_dict_keys])}的JSON形式回复."
    #     prompt = PromptTemplate.from_template(verified_prompt)
    #     chain=({"question": RunnablePassthrough()} | prompt | model | StrOutputParser())
    #     res=chain.invoke(question)
    #     print(res)
    #     try:
    #         json_part = find_and_parse_json(res)
    #         return json.dumps(json_part)
    #     except:
    #         return None
        
    # def out(question):
    #     prompt= f"你的任务是从以下的文本中提取疾病和症状：{question}"

    #     model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    #     struct_dict_keys=["疾病", "症状"]
    #     res=structify_output(model, prompt, question, struct_dict_keys)
    #     return res

    # from langchain_openai import ChatOpenAI
    # import os
    # os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
    # os.environ["OPENAI_API_KEY"] = 'sk-Gah5iODlG7OAojOK569f1103E7F9428c96E9Dc9e5954064a'
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

    from InterLM2 import InternLM2_20b
    from src.config.InternLM2 import internlm220bconf
    llm = InternLM2_20b(internlm220bconf)
    llm.load_model()

    graphragllm.load_model(llm)
    # res = graphragllm._call("我的症状是发热、呕吐，可能是什么病？")
    # res = graphragllm._call('一抽烟就恶心干呕是因为什么，一直觉得我自己的身体不是特别的舒服，最近就是时常头疼，症状很久了，不知道这种情况是咋回事，出现这种情况很频繁的。自己也挺烦恼的。')
    # res = graphragllm._call("奔豚病怎么治疗？")
    # res = graphragllm._call("'桂枝加桂汤方'的用法用量?")
    # res = graphragllm._call("麻黄连轺赤小豆汤的用法用量?")
    # res = graphragllm._call({"question":"你好，你是谁？"})
    # res = graphragllm._call({"question": "我还吐出脓血", "history":[("我肺不舒服，咳嗽得很严重，可能是什么病？", " ")]})
    # res = graphragllm._call("我还心烦，但不呕吐",[("我发热、呕吐，可能是什么病",'根据提供的参考上下文，结合症状发热和呕吐，可能的病因是百合病。')])
    # res = graphragllm._call({"question": "我的症状为咳嗽，吐出脓血，是什么问题？"})
    # res = graphragllm._call({"question": "胆石症的治疗应注意什么？"})
    res = graphragllm._call({"question": "我最近头痛、发热，还身体强硬，有什么病吗？"})
    print(res)
    # {'reference': {'structured_list': {'疾病': None, '症状': ['头痛', '发热', '身体强硬'], '方剂': None}, 'structured_data': (['腹满寒疝宿食病 - 临床表现 -> 头痛', '太阳病 - 临床表现 -> 头痛', '阳明病 - 临床表现 -> 头痛', '头痛头疯 - 临床表现 -> 头痛', '腹满寒疝宿食病 - 临床表现 -> 发热', '百合病 - 临床表现 -> 发热', '太阳病 - 临床表现 -> 发热', '伤寒 - 临床表现 -> 发热', '时疫喉病 - 临床表现 -> 发热', '太阳病 - 临床表现 -> 身体强硬'], {'腹满寒疝宿食病': '2/10', '太阳病': '3/5', '阳明病': '1/10', '头痛头疯': '1/1', '百合病': '1/5', '伤寒': '1/10', '时疫喉病': '1/3'}), 'unstructured_data': [Document(page_content='<目录>伤寒家秘的本卷之二\n\n<篇名>头痛\n\n属性：有头痛的情况，是寒邪进入足太阳经，向上攻到头部，这是表证。头痛，脉象浮紧，没有汗而且怕冷，可以发汗。头痛，脉象浮缓，有汗且怕冷，适宜解肌，按照前面所说的时令用药。患阳明病，不害怕寒冷反而害怕炎热，五六天没有大便，胃中实邪、燥热而口渴，热气向上攻到头目，脉象坚实的，用调胃承气汤攻下。有少阳头痛的，用小柴胡汤调和。湿邪为患导致鼻塞头痛的，用瓜蒂散搐鼻，黄水流出就会痊愈。因痰涎导致头痛，胸部胀满伴有寒热的，用瓜蒂散催吐。厥阴病干呕吐出涎沫、头痛的，用吴茱萸汤主治。三阳虽然都有头痛，但不像太阳经专门主管头痛。三阴没有头痛，只有厥阴有头痛，是因为其脉系联络在头顶。如果头痛牵连到胸部，手足都发青，这是真头痛，一定会死亡。\n\n', metadata={'source': '/home/ubuntu/PuyuanChallenge/Dist/src/data/retrieve_data/470-伤寒六书.txt', 'start_index': 36809, 'relevance_score': 0.5252139568328857})]}, 'question': {'question': '我最近头痛、发热，还身体强硬，有什么病吗？'}, 'answer': '根据您提供的症状，头痛、发热和身体强硬，结合可参考的上下文，可能与以下疾病有关：\n\n1. 太阳病：太阳病的主要症状包括头痛、发热、身体强硬等。根据上下文，太阳病的脉象浮紧，没有汗且怕冷，可以使用发汗的方法进行治疗。\n\n2. 阳明病：阳明病的主要症状包括头痛、发热、身体强硬等。根据上下文，阳明病的脉象浮缓，有汗且怕冷，适宜解肌，按照前面所说的时令用药。\n\n3. 伤寒：伤寒的主要症状包括头痛、发热、身体强硬等。根据上下文，伤寒的脉象坚实的，可以使用调胃承气汤攻下。\n\n4. 时疫喉病：时疫喉病的主要症状包括头痛、发热、身体强硬等。根据上下文，时疫喉病的脉象坚实的，可以使用瓜蒂散搐鼻。\n\n建议您及时就医，由专业医生进行诊断和治疗。同时，注意保持良好的生活习惯，保持充足的睡眠和饮食均衡，避免过度劳累和情绪波动，有助于恢复健康。'}