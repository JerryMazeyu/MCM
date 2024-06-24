from abc import ABC
from src.utils.io import LoaderMixin

class Context(ABC, LoaderMixin):
    """Context of action
    """
    def __init__(self, contents):
        self._load(contents)
    
    @property
    def content(self):
        return vars(self)

def register_action(method):
    """
    Decorator to capture the return value of a method and assign it to self.context.

    Parameters:
        method (callable): The method to be decorated.

    Returns:
        callable: The wrapper function that enhances the original method.
    """
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if not isinstance(result, dict):
            raise ValueError("Make sure that you return a dict.")
        self.context = Context(result)
        return result
    return wrapper
