from data.dataset import CIFAR10GrayToColor
from models.model import get_model
from training.train import train_model
from configs.config import config
from utils.transforms import get_transforms
from torch.utils.data import DataLoader, Subset

transform, _ = get_transforms()
train_dataset = CIFAR10GrayToColor(root="./data", train=True, transform=transform)
train_subset = Subset(train_dataset, range(config["train_subset_size"]))
train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)

model = get_model(config)
train_model(model, train_loader, config)
