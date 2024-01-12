from contrastive.model_wrappers.registry import ModelWrapperBase, ModelWrapperRegister
from contrastive.models.vicreg.resnet import resnet50, resnet50x2, resnet200x2
from ..utils import val_transform, remove_prefix

_vicreg_checkpoint_path = {
    'resnet50': '/mnt/data/user/tc_ai/user/songjunjie/checkpoints/vicreg/resnet50.pth',
    'resnet50x2': '/mnt/data/user/tc_ai/user/songjunjie/checkpoints/vicreg/resnet50x2.pth',
    'resnet200x2': '/mnt/data/user/tc_ai/user/songjunjie/checkpoints/vicreg/resnet200x2.pth'
}

class VICRegWrapperBase(ModelWrapperBase):
    def __init__(self, device='cuda', checkpoint_path=None):
        super().__init__()
        self.model = self._build_model()
        self.model.to(device)
        if checkpoint_path is not None:
            self.overlay_state_dict(checkpoint_path)

        self.device = device

    @property
    def transform(self):
        return val_transform
    
    def encode_image(self, image):
        image = image.to(self.device)
        out = self.model(image)
        return out

    def _build_model(self):
        raise NotImplementedError


@ModelWrapperRegister.register('vicreg-resnet50')
class VICRegResNet50(VICRegWrapperBase):
    def __init__(checkpoint_path, **kwargs):
        super().__init__(
            checkpoint_path=_vicreg_checkpoint_path['resnet50'],
            **kwargs
            )

    def _build_model(self):
        model, ndim = resnet50()
        return model
        
@ModelWrapperRegister.register('vicreg-resnet50x2')
class VICRegResNet50x2(VICRegWrapperBase):
    def __init__(checkpoint_path, **kwargs):
        super().__init__(
            checkpoint_path=_vicreg_checkpoint_path['resnet50x2'],
            **kwargs
            )

    def _build_model(self):
        model, ndim = resnet50x2()
        return model

@ModelWrapperRegister.register('vicreg-resnet200x2')
class VICRegResNet200x2(VICRegWrapperBase):
    def __init__(checkpoint_path, **kwargs):
        super().__init__(
            checkpoint_path=_vicreg_checkpoint_path['resnet200x2'],
            **kwargs
            )

    def _build_model(self):
        model, ndim = resnet200x2()
        return model