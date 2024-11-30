import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

from PIL import Image
import numpy as np


class CIFAR10GrayToColor(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        gray_image = Image.fromarray(np.array(image)).convert('L')

        if self.transform:
            gray_image = self.transform(gray_image)
            image = self.transform(image)

        return gray_image, image
