from abc import ABC, abstractmethod
from langchain_core.language_models.llms import LLM



class BaseLLM(LLM, ABC):
    """Base LLM abstract class
    """

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load model explicitly to save the resource.
        """
        pass
    
    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Model nickname
        """
        return "Base"