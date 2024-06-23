from typing import Union, Any, Dict
from src.utils.utils import log
import os

class ShowConfigMixin:
    """Show configs of the class
    """

    def _show(self):
        if hasattr(self, 'LOGINFO'):
            pass
        attributes = vars(self)
        s = ''
        s += f"Attributes of {self.__class__.__name__}: "
        for key, value in attributes.items():
            s += f"{key} = {value}; "
        log(s)

class LoaderMixin:
    """Load properties from another class or a dictionary into the current instance.
    """
    def _load(self, cfg:Union[Dict[str, Any], Any]):
        if isinstance(cfg, dict):
            for k,v in cfg.items():
                setattr(self, k, v)
        else:
            for k,v in vars(cfg).items():
                setattr(self, k, v)

class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



