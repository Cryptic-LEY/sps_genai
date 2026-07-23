import io

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10, z_dim=100):
    """Generate num_samples images from random latent vectors and display them on a grid."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        images = model(z).cpu()

    images = (images + 1) / 2  # rescale from Tanh's [-1, 1] to [0, 1] for display
    grid = make_grid(images, nrow=int(num_samples ** 0.5) or 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


def generate_image_grid_png(model, device, num_samples=10, z_dim=100):
    """Generate num_samples images, arrange them on a grid, and return PNG bytes (for serving via an API)."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        images = model(z).cpu()

    images = (images + 1) / 2
    grid = make_grid(images, nrow=int(num_samples ** 0.5) or 1)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


def langevin_dynamics(energy_model, inp_imgs, steps, step_size, noise_std):
    """Run Langevin dynamics: gradient descent on the *input image* (not the model
    weights) to lower its energy under energy_model. Matches Practical 1's
    generate_samples. Used both to refresh the EBM training replay buffer and to
    sample from scratch at inference time.
    """
    energy_model.eval()

    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)
        energy = energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


def generate_ebm_grid_png(model, device, num_samples=10, image_size=32, num_channels=3, steps=256, step_size=10.0, noise_std=0.005):
    """Sample num_samples images from random noise via Langevin dynamics, arrange them
    on a grid, and return PNG bytes."""
    init_imgs = torch.rand((num_samples, num_channels, image_size, image_size), device=device) * 2 - 1
    samples = langevin_dynamics(model, init_imgs, steps=steps, step_size=step_size, noise_std=noise_std)

    images = ((samples + 1) / 2).clamp(0.0, 1.0).cpu()
    grid = make_grid(images, nrow=int(num_samples ** 0.5) or 1)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


def reverse_diffusion(model, schedule_fn, initial_noise, diffusion_steps):
    """Iteratively denoise initial_noise into an image, matching Practical 2's
    reverse_diffusion. Used both to preview training progress and to sample at
    inference time."""
    device = initial_noise.device
    step_size = 1.0 / diffusion_steps
    current_images = initial_noise
    pred_images = initial_noise

    with torch.no_grad():
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.size(0), 1, 1, 1), device=device) * (1 - step * step_size)
            noise_rates, signal_rates = schedule_fn(t)

            pred_noises = model(current_images, noise_rates ** 2)
            pred_images = (current_images - noise_rates * pred_noises) / signal_rates

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

    return pred_images


def generate_diffusion_grid_png(model, schedule_fn, device, num_samples=10, image_size=64, num_channels=3, diffusion_steps=20):
    """Sample num_samples images from Gaussian noise via reverse diffusion, arrange
    them on a grid, and return PNG bytes."""
    model.eval()
    initial_noise = torch.randn((num_samples, num_channels, image_size, image_size), device=device)
    samples = reverse_diffusion(model, schedule_fn, initial_noise, diffusion_steps)

    images = ((samples + 1) / 2).clamp(0.0, 1.0).cpu()
    grid = make_grid(images, nrow=int(num_samples ** 0.5) or 1)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf
