from typing import Union, Any, Dict
from src.utils.utils import log

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





