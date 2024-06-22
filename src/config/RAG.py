from src.config.base import RAGConfig
from src.config.base import GraphRAGConfig

class RAGC(RAGConfig):
    def __init__(self):
        # self.model_path = "/home/ubuntu/PuyuanChallenge/3rdParty/GithubRepos/Qilin-Med-VL-Chat-model"
        # self.image_aspect_ratio = "pad"
        # self.conv_mode = "llava_v1"
        # self.temperature = 0.2
        # self.max_new_tokens = 512
        # self.bit = "Full"  # "Full \ 8bit \ 4bit"
        # self.device = "cuda"
        # self.device_map = "auto"

        self.embeddings_path="/home/ubuntu/PuyuanChallenge/Dist/src/backend/sentence-transformer"
        self.persist_directory = '/home/ubuntu/PuyuanChallenge/Dist/src/data/retrieve_vector'
        self.bm25retriever_path='/home/ubuntu/PuyuanChallenge/Dist/src/backend/bm25retriever/bm25retriever.pkl'
        self.reranker_args = {'model': '/home/ubuntu/PuyuanChallenge/Dist/src/backend/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:1'}

# ragconf = RAGC()
# graphragconf = GraphRAGConfig()

