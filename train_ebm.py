import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.utils import get_device

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    IMAGE_SIZE = 32
    train_loader = get_data_loader(batch_size=128, train=True, image_size=IMAGE_SIZE, normalize=True)

    model = get_model("EBM")

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

    train_ebm(
        model,
        train_loader,
        optimizer,
        device=device,
        epochs=12,
        image_size=IMAGE_SIZE,
        num_channels=3,
        steps=60,
        step_size=10.0,
        noise=0.005,
        alpha=0.1,
        checkpoint_dir="checkpoints_ebm",
    )

    print("Finished Training")
