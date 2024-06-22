from src.config.RAG import RAGC
from src.backend.kernelModels.baseModel import BaseLLM
from src.utils.io import LoaderMixin
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pickle
from langchain.retrievers import EnsembleRetriever
import sys
sys.path.append('/home/ubuntu/PuyuanChallenge/Dist')
from libs.BCEmbedding.tools.langchain import BCERerank
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Extra


class RAG(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)

    def load_retriever(self):
    # 加载向量数据库
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_path)

        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=self.persist_directory  # 允许我们将persist_directory目录保存到磁盘上
        )

        chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 1})

        bm25retriever = pickle.load(open(self.bm25retriever_path, 'rb'))
        bm25retriever.k = 1
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, chroma_retriever], weights=[0.6, 0.4])
        
        # # 创建带reranker的检索器，对大模型过滤器的结果进行再排序
        # reranker_args = {'model': '/home/ubuntu/PuyuanChallenge/Dist/src/backend/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:1'}
        reranker = BCERerank(**self.reranker_args)
        # compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)
        compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=chroma_retriever)

        return compression_retriever
    
    def load_chain(self, llm, verbose=False):
        # 定义Prompt Template
        template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        请提供详细而清晰的回答。确保回答涵盖相关医疗知识和实际诊断，尽量详细回答问题，并尽量避免简单带过问题。
        可参考的上下文：
        ···
        {context}
        ···
        问题: {question}
        有用的回答:"""

        rag_prompt = PromptTemplate.from_template(template)

        retriever = self.load_retriever()

        # 不返回source
        # qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True, chain_type="stuff",
        #                                         chain_type_kwargs={"prompt": rag_prompt, "verbose": verbose})

        # 返回source
        rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source 
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def load_model(self, llm):
        self.model = self.load_chain(llm)
    
    def _call(self, prompt, history=[]):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            response = self.model.invoke(prompt)
            # ['result']
            return response

    @property
    def _llm_type(self):
        return "RAGLLM"
    


if __name__ == '__main__':
    ragconf = RAGC()
    ragllm = RAG(ragconf)
    unstructured_retriever=ragllm.load_retriever()
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    from langchain_openai import ChatOpenAI
    import os
    os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
    os.environ["OPENAI_API_KEY"] = 'sk-Gah5iODlG7OAojOK569f1103E7F9428c96E9Dc9e5954064a'
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

    # from InterLM2 import InternLM2_20b
    # from src.config.InternLM2 import internlm220bconf
    # llm = InternLM2_20b(internlm220bconf)
    # llm.load_model()

    ragllm.load_model(llm)
    res= ragllm._call("卵巢早衰怎么办？")
    print(res)

    # retr=ragllm.load_retriever()
    # res=retr.invoke("卵巢早衰怎么办？")
    # print(res)


    # gpt3.5
    # {'context': [Document(page_content='\n{"问": "卵巢早衰应该咋治恢复快，最近经常出现经量减少，经期缩短，月经周期增长，经检查是卵巢早衰，请问卵巢早衰能彻底恢复吗", "答": "卵巢早衰能彻底恢复。卵巢早衰能彻底恢复的，患者饮食方面要留意营养平衡，除了蛋白质足量摄取外，脂肪及糖类应足量，同时特别注意维生素E、D及矿物质如铁钙的消化，其中适当消化维生素E可以彻底清除自由基，稳定皮肤弹性，延后性腺膨胀的进程，起些抗衰老的作用，并可调整免疫功能，每日150—300毫克即可。要适当增强活动，活动有助于增进新陈代谢及血液循环，减缓器官衰老。活动应当量力而行持之以恒，循序渐进，如跑步、散步、广播操、太极拳均是较适宜的活动。"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 1537, 'relevance_score': 0.6189717054367065}),
    #             Document(page_content='{"问": "背痛的放射治疗有些什么？", "答": "单纯体外放射治疗"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 15374, 'relevance_score': 0.4013370871543884})], 
    #  'question': '卵巢早衰怎么办？', 
    #  'answer': '卵巢早衰的治疗可以采取多种方式。
    #  首先，医生可能会建议调整饮食，确保摄取足够的营养，包括蛋白质、脂肪、糖类以及维生素和矿物质，特别是维生素E、D以及铁和钙等矿物质，这有助于促进身体的健康恢复和代谢调节。
    #  此外，适当的运动也非常重要，可以选择适合自己的轻度运动，如散步、太极拳等，有助于增加新陈代谢和血液循环，减缓器官衰老的进程。
    #  在治疗过程中，医生可能会根据具体情况开具药物治疗，如激素替代疗法等，以帮助调节激素水平和改善症状。
    #  对于每位患者来说，治疗方案可能会有所不同，建议在专业医生的指导下进行个性化的治疗方案制定和调整。'} []

    # Internlm2-20b
    # {'context': [Document(page_content='\n{"问": "卵巢早衰应该咋治恢复快，最近经常出现经量减少，经期缩短，月经周期增长，经检查是卵巢早衰，请问卵巢早衰能彻底恢复吗", "答": "卵巢早衰能彻底恢复。卵巢早衰能彻底恢复的，患者饮食方面要留意营养平衡，除了蛋白质足量摄取外，脂肪及糖类应足量，同时特别注意维生素E、D及矿物质如铁钙的消化，其中适当消化维生素E可以彻底清除自由基，稳定皮肤弹性，延后性腺膨胀的进程，起些抗衰老的作用，并可调整免疫功能，每日150—300毫克即可。要适当增强活动，活动有助于增进新陈代谢及血液循环，减缓器官衰老。活动应当量力而行持之以恒，循序渐进，如跑步、散步、广播操、太极拳均是较适宜的活动。"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 1537, 'relevance_score': 0.6189717054367065}), 
    #  Document(page_content='{"问": "背痛的放射治疗有些什么？", "答": "单纯体外放射治疗"}', metadata={'source': '/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/MedicalGPT/data/rag/medical_corpus.txt', 'start_index': 15374, 'relevance_score': 0.4013370871543884})], 
    # 'question': '卵巢早衰怎么办？', 
    # 'answer': '卵巢早衰可以通过饮食调理、适当运动、药物治疗等方式来缓解症状。饮食方面要留意营养平衡，除了蛋白质足量摄取外，脂肪及糖类应足量，同时特别注意维生素E、D及矿物质如铁钙的消化，其中适当消化维生素E可以彻底清除自由基，稳定皮肤弹性，延后性腺膨胀的进程，起些抗衰老的作用，并可调整免疫功能，每日150—300毫克即可。要适当增强活动，活动有助于增进新陈代谢及血液循环，减缓器官衰老。活动应当量力而行持之以恒，循序渐进，如跑步、散步、广播操、太极拳均是较适宜的活动。药物治疗方面，可以口服一些维生素E、D等药物，也可以口服一些中药进行调理，如当归、熟地、何首乌等。如果症状严重，可以考虑进行激素替代治疗，但需要在医生的指导下进行。'}