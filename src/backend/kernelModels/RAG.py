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
    


