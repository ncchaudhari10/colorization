import torch
from tqdm import tqdm

def infer(model, test_loader, scheduler, config):

    model.eval()
    gray_images, original_images, samples = [], [], []
    device = config["device"]

    with torch.no_grad():
        for gray, original in tqdm(test_loader, desc="Inference Progress"):
            gray, original = gray.to(device), original.to(device)

            # Start from pure noise
            batch_size = gray.size(0)
            noisy_images = torch.randn((batch_size, 3, config["image_size"], config["image_size"])).to(device)

            # Perform denoising with tqdm for reverse diffusion steps
            for t in tqdm(range(scheduler.config.num_train_timesteps - 1, -1, -1),
                          desc="Reverse Diffusion",
                          leave=False,
                          total=scheduler.config.num_train_timesteps):
                timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
                x_t = torch.cat((noisy_images, gray), dim=1)
                noise_pred = model(x_t, timesteps).sample
                noisy_images = scheduler.step(noise_pred, t, noisy_images).prev_sample

            samples.extend(noisy_images.cpu())  # Move to CPU for visualization/storage
            gray_images.extend(gray.cpu())
            original_images.extend(original.cpu())

    return samples, gray_images, original_images
