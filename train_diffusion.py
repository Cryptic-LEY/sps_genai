import copy

import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from helper_lib.data_loader import get_data_loader
from helper_lib.diffusion import offset_cosine_diffusion_schedule
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    IMAGE_SIZE = 64
    BATCH_SIZE = 64
    # CPU training is slow at 64x64; a subset keeps a full run to a reasonable
    # wall-clock time while still giving the UNet a varied training signal.
    SUBSET_SIZE = 8000

    full_loader = get_data_loader(batch_size=BATCH_SIZE, train=True, image_size=IMAGE_SIZE, normalize=True)
    subset = Subset(full_loader.dataset, range(SUBSET_SIZE))
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model("Diffusion")
    ema_model = copy.deepcopy(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_diffusion(
        model,
        ema_model,
        train_loader,
        optimizer,
        schedule_fn=offset_cosine_diffusion_schedule,
        device=device,
        # Practical 2 trains for 50 epochs on GPU; 12 keeps CPU training under
        # an hour on the 8000-image subset above while still exercising the
        # full pipeline end-to-end.
        epochs=12,
        ema_decay=0.8,
        checkpoint_dir="checkpoints_diffusion",
    )

    print("Finished Training")
