import bs4
from langchain import hub
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from langchain_core.runnables import RunnableParallel
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from LLMAgent import LLMAgent
from src.backend.kernelModels.InterLM2 import internlm220b
from langchain_community.vectorstores import FAISS


import torch
import sys
sys.path.append('/home/ubuntu/PuyuanChallenge/Dist')
from libs.BCEmbedding.tools.langchain import BCERerank

os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
os.environ["OPENAI_API_KEY"] = 'sk-Gah5iODlG7OAojOK569f1103E7F9428c96E9Dc9e5954064a'

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]="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"

from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
# from BCEmbedding.tools.langchain import BCERerank
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.runnables import RunnableParallel


reranker_args = {'model': '/home/ubuntu/PuyuanChallenge/Dist/src/backend/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:1'}
reranker = BCERerank(**reranker_args)


def load_retriever():
    # 加载向量数据库
    embeddings = HuggingFaceEmbeddings(model_name="/home/ubuntu/PuyuanChallenge/Dist/src/backend/sentence-transformer")
    persist_directory = '/home/ubuntu/PuyuanChallenge/Dist/src/data/retrieve_vector'
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )

    chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    bm25retriever = pickle.load(open('/home/ubuntu/PuyuanChallenge/Dist/src/backend/bm25retriever/bm25retriever.pkl', 'rb'))
    bm25retriever.k = 2

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, chroma_retriever], weights=[0.4, 0.6])
    
    # # 创建带reranker的检索器，对大模型过滤器的结果进行再排序
    reranker_args = {'model': '/home/ubuntu/PuyuanChallenge/Dist/src/backend/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:1'}
    reranker = BCERerank(**reranker_args)
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)
    # bce_reranker_config = load_config('rag_langchain', 'bce_reranker_config')
    # reranker = BCERerank(**bce_reranker_config)
    # # 依次调用ensemble_retriever与reranker，并且可以将替换假设问题为原始菜谱的Retriever
    # compression_retriever = HyQEContextualCompressionRetriever(base_compressor=reranker,
    #                                                            base_retriever=ensemble_retriever)
    return compression_retriever

a=load_retriever()


def load_chain(llm, verbose=False):
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

    retriever = load_retriever()

    # 不返回source
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True, chain_type="stuff",
                                            chain_type_kwargs={"prompt": rag_prompt, "verbose": verbose})

    # 返回source
    rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | rag_prompt
    | llm
    | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source 


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain_instance = None

# @torch.inference_mode()
def generate_interactive_rag(
        llm,
        question,
        verbose=False
):
    global chain_instance
    if chain_instance is None:
        chain_instance = load_chain(llm, verbose=verbose)
    return chain_instance({"query": question})['result']


llm = LLMAgent(internlm220b)
question="你是谁"
result=generate_interactive_rag(llm, question)
print(result)




