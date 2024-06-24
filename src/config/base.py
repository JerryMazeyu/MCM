from src.utils.io import ShowConfigMixin


class BaseLLMConfig(ShowConfigMixin):
    def __init__(self):
        self.model = ''
        self.tokenizer = ''
        self.weights = ''


class BaseVLMConfig(ShowConfigMixin):
    def __init__(self):
        self.model = ''
        self.tokenizer = ''

class RAGConfig(ShowConfigMixin):
    def __init__(self):
        self.model = ''
        self.tokenizer = ''

class TemplateConfig(ShowConfigMixin):
    def __init__(self):
        pass

class GraphRAGConfig(ShowConfigMixin):
    def __init__(self):
        pass

__all__ = ["BaseLLMConfig", "BaseVLMConig", "RAGConfig", "TemplateConfig", "GraphRAGConfig"]