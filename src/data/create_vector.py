from langchain_community.document_loaders  import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from tqdm import tqdm
import pickle

#加载retrieve_data下的文档
# loader = ('/home/ubuntu/PuyuanChallenge/Dist/src/data/retrieve_data')

rag_path ="/home/ubuntu/PuyuanChallenge/Dist/src/data/retrieve_data"
loader = DirectoryLoader(rag_path)
print("Loading documents...")
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

# 分词
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, add_start_index=True, separators=["<目录>"])
print("Splitting documents...")
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks.")

# for i in range(10):
#     print('第i个: ',split_docs[i].page_content)

# 加载开源词向量模型，已下载到本地
embeddings = HuggingFaceEmbeddings(model_name="/home/ubuntu/PuyuanChallenge/Dist/src/backend/sentence-transformer")

# 加载数据库
persist_directory = '/home/ubuntu/PuyuanChallenge/Dist/src/data/retrieve_vector'
print("Creating vector store...")
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
print("Vector store created.")

# 将加载的向量数据库持久化到磁盘上
print("Persisting vector store to disk...")
vectordb.persist()
print("Vector store persisted.")

# 创建BM25检索器
print("Creating BM25 retriever...")
bm25retriever = BM25Retriever.from_documents(split_docs)
bm25retriever.k =  2

# BM25Retriever序列化到磁盘
bm25retriever_path='/home/ubuntu/PuyuanChallenge/Dist/src/backend/bm25retriever'
if not os.path.exists(bm25retriever_path):
    os.mkdir(bm25retriever_path)
    
print("Serializing BM25 retriever...")
pickle.dump(bm25retriever, open('/home/ubuntu/PuyuanChallenge/Dist/src/backend/bm25retriever/bm25retriever.pkl', 'wb'))
print("BM25 retriever serialized.")


# chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, chroma_retriever], weights=[0.4, 0.6])

