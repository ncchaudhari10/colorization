import torch
from diffusers import UNet2DModel

def get_model(config):
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=4,
        out_channels=3,
        layers_per_block=4,
        block_out_channels=config["block_out_channels"],
    )
    return model.to(config["device"])
