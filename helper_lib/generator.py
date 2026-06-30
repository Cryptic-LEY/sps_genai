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
