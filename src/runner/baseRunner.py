from abc import ABC, abstractclassmethod
from typing import Any, List

class BaseRunner(ABC):
    
    @abstractclassmethod
    def run(self):
        pass