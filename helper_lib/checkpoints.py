import os

import torch


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints", filename=None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }

    filename = filename or f"epoch_{epoch:03d}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]
    return epoch, loss, accuracy
