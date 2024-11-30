import os

import torch
from diffusers import DDPMScheduler
from torch.nn import MSELoss
from torch.optim import AdamW
from tqdm import tqdm


def train_model(model, train_loader, config):
    # Prepare training components
    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"]
    )
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = MSELoss()
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        losses = 0
        for batch in progress_bar:
            gray, color = batch
            gray, color = gray.to(config["device"]), color.to(config["device"])

            # Add noise
            noise = torch.randn_like(color)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (gray.size(0),)).long().to(
                config["device"])
            noisy_images = scheduler.add_noise(color, noise, timesteps)

            # Concatenate grayscale and noisy color images
            x_t = torch.cat((noisy_images, gray), dim=1)

            # Predict noise
            noise_pred = model(x_t, timesteps).sample

            # Compute loss
            loss = criterion(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            losses += loss.item()
            # Update progress bar
            progress_bar.set_postfix({"loss": losses})

        # Save model after each epoch
    torch.save(model.state_dict(), os.path.join(save_dir, f"colorization_epoch.pth"))

    print(f"Training completed. Models are saved in {save_dir}")
