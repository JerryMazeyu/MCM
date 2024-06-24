import os
import argparse
import yaml
from src.utils.utils import log
import warnings
warnings.filterwarnings('ignore')
# from src.backend.kernelModels.InterLM2 import InternLM2_20b
# from src.runner.diagnoseRunner import DiagnoseRunner

def add_prefix_to_path_values(d, prefix="", path_separator="/"):
    for key, value in d.items():
        if isinstance(value, str) and path_separator in value:
            prefixed_path = os.path.join(prefix, value)
            d[key] = prefixed_path
        elif isinstance(value, dict):
            add_prefix_to_path_values(value, prefix, path_separator)


def load_conf(conf):
    root = conf['General']['root']
    add_prefix_to_path_values(conf, prefix=root)
    log("Loading Neo4j configuration...")
    # os.environ["NEO4J_URI"] = conf['Neo4j']['url']
    # os.environ["NEO4J_USERNAME"] = conf['Neo4j']['username']
    # os.environ["NEO4J_PASSWORD"] = conf['Neo4j']['password']
    
    if 'InternLM2_20b' in list(conf["LLMs"].keys()):
        log("Loading InternLM2_20b...")
        from src.config.InternLM2 import InternLM2_20bConfig
        internlm2_20bconfig = InternLM2_20bConfig()
        tmp = conf['LLMs']['InternLM2_20b']
        setattr(internlm2_20bconfig, 'version', tmp['version'])
        setattr(internlm2_20bconfig, 'weights_dict', tmp['weights_dict'])
        setattr(internlm2_20bconfig, 'tokenizer_name', tmp['weights_dict'][tmp['version']])
        setattr(internlm2_20bconfig, 'device', tmp['device'])
        internlm2_20bconfig._show()
        from src.backend.kernelModels.InterLM2 import InternLM2_20b
        llm = InternLM2_20b(internlm2_20bconfig)
    else:
        raise ValueError("Sorry, only support InternLM2_20b now.")
    
    if 'MedDr' in list(conf["VLMs"].keys()):
        log("Loading MedDr...")
        from src.config.MedDr import MedDrConfig
        meddrconfig = MedDrConfig()
        tmp = conf['VLMs']['MedDr']
        setattr(meddrconfig, 'version', tmp['version'])
        setattr(meddrconfig, 'weights_dict', tmp['weights_dict'])
        setattr(meddrconfig, 'tokenizer_name', tmp['weights_dict'][tmp['version']])
        setattr(meddrconfig, 'device', tmp['device'])
        meddrconfig._show()
        from src.backend.kernelModels.MedDr import MedDr
        vlm = MedDr(meddrconfig)
    else:
        log("Only support MedDr now, we will load mock vlm.", level="WARNING")
    
    if 'RAG' in list(conf["RAGs"].keys()):
        from src.config.RAG import RAGC, GraphRAGConfig
        log("Loading RAG...")
        ragc = RAGC()
        tmp = conf['RAGs']['RAG']
        setattr(ragc, 'embeddings_path', tmp['embeddings_path'])
        setattr(ragc, 'persist_directory', tmp['persist_directory'])
        setattr(ragc, 'bm25retriever_path', tmp['bm25retriever_path'])
        setattr(ragc, 'reranker_args', tmp['reranker_args'])
        ragc._show()
        from src.backend.kernelModels.RAG import RAG
        rag = RAG(ragc)
        retriever=rag.load_retriever()
        
    if 'GraphRAG' in list(conf["RAGs"].keys()):
        log("Loading GraphRAG...")
        graphragc = GraphRAGConfig()
        tmp = conf['RAGs']['GraphRAG']
        backend = tmp['backend']
        choices = tmp['choice'].split(',')
        setattr(graphragc, 'backend', backend)
        setattr(graphragc, 'neo4j', tmp['Neo4j'])
        assert backend in choices, ValueError("Only support GPT4o & InternLM2 as backend.")
        if backend == 'gpt-4o':
            os.environ["HTTP_PROXY"] = tmp['gpt-4o']['http_proxy']
            os.environ["HTTPS_PROXY"] = tmp['gpt-4o']['http_proxy']
            os.environ["OPENAI_API_KEY"] = tmp['gpt-4o']['apikey']
            setattr(graphragc, 'gpt4o', tmp['gpt-4o'])
            from langchain_openai import ChatOpenAI
            gptllm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        else:
            log("Use gpt4o please.")
        graphragc._show()
        from src.backend.kernelModels.GraphRAG import GraphRAG
        graphrag = GraphRAG(graphragc, retriever)
    
    log("Loading Agent...")
    from src.agents.LLMAgent import LLMAgent
    from src.agents.MedDrAgent import MedDrAgent
    from src.agents.GraphRagAgent import GraphRagAgent
    from src.agents.diagnoseAgent import DiagnoseAgent
    llmagent = LLMAgent(llm)
    vlmagent = MedDrAgent(vlm)
    ragagent = GraphRagAgent(graphrag, gptllm)
    dagent = DiagnoseAgent(llmagent, vlmagent, ragagent)
    return dagent
    
    
        
    

def main():
    parser = argparse.ArgumentParser(description='Configure')
    parser.add_argument("-c", "--config", default="Dist/conf.yaml", help="Configure file (yaml) path.")

    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        conf = yaml.safe_load(file)

    dagent = load_conf(conf)
    from src.runner.diagnoseRunner import DiagnoseRunner
    runner = DiagnoseRunner(dagent)
    runner.run()

if __name__ == "__main__":
    main()
