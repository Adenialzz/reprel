from .registry import ModelWrapperRegister

def create_model(model_name, device='cuda', checkpoint=None):
    model = ModelWrapperRegister.build_model(model_name)
    return model

def list_models():
    return ModelWrapperRegister.list_models()

from .vicreg import *