import os
import os.path as osp
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from functools import partial
from songmisc.utils import send_to_device
# import sys; sys.path.append('/home/jeeves/JJ_Projects/songjunjie/ZCLIP')
# from src.CLIPWrappers import create_model

class Wrapper:
    def __init__(self, model_id='facebook/dinov2-base', device='cuda'):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = device
        self.model.to(self.device)
    
    @property
    def transform(self):
        return partial(self.processor, return_tensors="pt")
    
    def encode_image(self, inputs):
        inputs = {'pixel_values': inputs.pixel_values.squeeze().to(self.device)}
        # send_to_device(inputs, self.device)
        outputs = self.model(**inputs)
        cls_embedding = outputs.pooler_output
        return cls_embedding

# model = Wrapper(model_id='facebook/dinov2-base')

# model = create_model('open-clip', 'ViT-B-16@openai', device='cuda')
# model.overlay_state_dict('/mnt/data/user/tc_agi/multi_modal/checkpoints/clip-training/vit-b-16-st3-ep4.pt')
from .model_wrappers import create_model, list_models
# model = VICRegWrapper(device='cuda', checkpoint_path='/home/jeeves/JJ_Projects/github/analysis/checkpoints/vicreg/resnet50.pth')
print(list_models())
model = create_model('vicreg-resnet200x2')

data_root = '~/data/imagenet/train'
batch_size = 128
save_dir = "./vicreg_renset50"
os.makedirs(save_dir, exist_ok=True)
dataset_type = 'imagefolder'
if dataset_type == 'imagefolder':
    dataset = torchvision.datasets.ImageFolder(data_root, model.transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

num_train_samples = 40000
num_test_samples = 10000

train_features, train_labels = torch.Tensor([]), torch.Tensor([])
test_features, test_labels = torch.Tensor([]), torch.Tensor([])
with torch.no_grad():
    for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        feat = model.encode_image(image).cpu()
        if i < num_train_samples / batch_size:
            train_labels = torch.cat([train_labels, label], dim=0)
            train_features = torch.cat([train_features, feat], dim=0)
        elif i < (num_train_samples + num_test_samples) / batch_size:
            test_labels = torch.cat([test_labels, label], dim=0)
            test_features = torch.cat([test_features, feat], dim=0)
        else:
            break

np.save(osp.join(save_dir, 'train_features.npy'), train_features)
np.save(osp.join(save_dir, 'train_labels.npy'), train_labels)
np.save(osp.join(save_dir, 'test_features.npy'), test_features)
np.save(osp.join(save_dir, 'test_labels.npy'), test_labels)

