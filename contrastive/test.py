import torch
from contrastive.model_wrappers.base_wrapper import VICRegWrapper


model = VICRegWrapper(device='cuda')

image = torch.randn(4, 3, 224, 224)
feat = model.encode_image(image)
print(feat.shape)
