import torch
from typing import Dict
from contrastive.utils import remove_prefix

class ModelWrapperBase:
    def __init__(self, device='cuda', checkpoint_path=None):
        self.model = self._build_model()

    def overlay_state_dict(self, checkpoint_path):
        print(f"overlay model with state dict at {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location='cpu')
        if hasattr(sd, 'state_dict'):
            sd = sd['state_dict']

        sd = remove_prefix(sd)
        self.model.load_state_dict(sd)
    
    def _build_model(self):
        raise NotImplementedError
    
    def encode_image(self):
        raise NotImplementedError


class ModelWrapperRegister:
    _registry: Dict[str, ModelWrapperBase] = {}

    @classmethod
    def register(cls, name):
        def wrap(cc):
            if name in cls._registry:
                raise ValueError(f"exist registered with name: {name}")
            cls._registry[name] = cc
            return cc
        return wrap

    @classmethod
    def build_model(self, name):
        return self._registry[name]()
    
    @classmethod
    def list_models(self):
        return list(self._registry.keys())
    
