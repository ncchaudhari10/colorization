import torchvision.transforms as transforms


def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1),
    ])
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(-1, 1)),
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.Lambda(lambda x: x * 255),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        transforms.Lambda(lambda x: x.byte()),
    ])
    return transform, reverse_transform
