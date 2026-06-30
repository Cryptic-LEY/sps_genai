import os

import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            progress.set_postfix(
                {
                    "loss": f"{running_loss / running_total:.4f}",
                    "acc": f"{running_correct / running_total:.3f}",
                }
            )

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / running_total
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        save_checkpoint(model, optimizer, epoch + 1, train_loss, train_accuracy, checkpoint_dir)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                checkpoint_dir=os.path.join(checkpoint_dir, "best"),
                filename="model.pth",
            )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%"
        )

    return model


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10, z_dim=100, checkpoint_dir="checkpoints"):
    """Adversarial training loop for a GAN.

    model: tuple (generator, discriminator), as returned by get_model("GAN")
    optimizer: tuple (optimizer_g, optimizer_d)
    """
    generator, discriminator = model
    optimizer_g, optimizer_d = optimizer

    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        running_loss_g, running_loss_d = 0.0, 0.0

        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for real_images, _ in progress:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()

            outputs_real = discriminator(real_images)
            loss_real = criterion(outputs_real, real_labels)

            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            optimizer_g.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()
            progress.set_postfix({"D loss": f"{loss_d.item():.4f}", "G loss": f"{loss_g.item():.4f}"})

        avg_loss_d = running_loss_d / len(data_loader)
        avg_loss_g = running_loss_g / len(data_loader)

        save_checkpoint(generator, optimizer_g, epoch + 1, avg_loss_g, 0.0, checkpoint_dir, filename="generator_latest.pth")
        save_checkpoint(discriminator, optimizer_d, epoch + 1, avg_loss_d, 0.0, checkpoint_dir, filename="discriminator_latest.pth")

        print(f"Epoch {epoch + 1}: D loss={avg_loss_d:.4f}, G loss={avg_loss_g:.4f}")

    save_checkpoint(generator, optimizer_g, epochs, avg_loss_g, 0.0, checkpoint_dir=f"{checkpoint_dir}/best", filename="generator.pth")
    save_checkpoint(discriminator, optimizer_d, epochs, avg_loss_d, 0.0, checkpoint_dir=f"{checkpoint_dir}/best", filename="discriminator.pth")

    return generator, discriminator
