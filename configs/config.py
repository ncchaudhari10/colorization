import torch

config = {
    "image_size": 32,
    "train_subset_size": 1000,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "num_epochs": 10,
    "num_train_timesteps": 1000,
    "beta_schedule": "linear",
    "block_out_channels": (64, 128, 256, 512),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_dir": "./saved_models",
    "test_subset_size":8
}
