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
from src.utils.io import suppress_stdout_stderr


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
            self.ori_model = ChatOpenAI(model="gpt-4o", temperature=0)
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
        struct_dict_keys = ["疾病", "症状", "方剂"]
        res=self.structify_output(self.ori_model, prompt, question, struct_dict_keys)
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
                | self.ori_model
                | StrOutputParser(),
            ),

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


    def str_reference(self, x):
        unstructured_str = "\n\n".join(doc.page_content for doc in x["unstructured_data"])
        final_data = f"""Structured data: {x["structured_data"]}. Unstructured data: {unstructured_str}"""
        return final_data

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def load_model(self, llm=None):
        if not llm:
            self.model = self.load_chain(self.ori_model)
        else:
            self.model = self.load_chain(llm)
    
    def _call(self, prompt):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            with suppress_stdout_stderr():
                response = self.model.invoke(prompt)
            # {"question": prompt, "history": history}
            # ['result']
            return response

    @property
    def _llm_type(self):
        return "GraphRAG"


