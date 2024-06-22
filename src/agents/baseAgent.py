from abc import ABC, abstractmethod
from typing import List

class BaseAgent(ABC):
    """Abstract method of agents
    """
    def __init__(self):
        self.context = None
        self.SYSINFO = ''
    
    @property
    @abstractmethod
    def actions(self)->List[str]:
        """Return all the actions.

        Returns:
            List[str]: Actions the agent has.
        """
        return []
    
    def execute(self, action: str, **kwargs):
        method = getattr(self, action, None)
        
        if not callable(method):
            raise AttributeError(f"Method {action} not found in {self.__class__.__name__}")
        
        return method(**kwargs)
    
    