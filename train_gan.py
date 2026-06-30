import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    Z_DIM = 100
    train_loader = get_mnist_loader(batch_size=64, train=True)

    generator, discriminator = get_model("GAN", z_dim=Z_DIM)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train_gan(
        (generator, discriminator),
        train_loader,
        criterion,
        (optimizer_g, optimizer_d),
        device=device,
        epochs=20,
        z_dim=Z_DIM,
        checkpoint_dir="checkpoints_gan",
    )

    print("Finished Training")
