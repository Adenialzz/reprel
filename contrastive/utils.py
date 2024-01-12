import torchvision.transforms as transforms

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def remove_prefix(state_dict, prefix='module.'):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
