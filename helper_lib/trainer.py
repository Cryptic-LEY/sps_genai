import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .evaluator import evaluate_model
from .generator import langevin_dynamics


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


class _EBMBuffer:
    """Replay buffer of Langevin samples, so sampling doesn't restart from random
    noise every batch. Matches Practical 1's Buffer class."""

    def __init__(self, model, device, image_size=32, num_channels=3, max_size=8192, pool_size=128):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.num_channels = num_channels
        self.max_size = max_size
        self.pool_size = pool_size
        self.examples = [
            torch.rand((1, num_channels, image_size, image_size), device=device) * 2 - 1
            for _ in range(pool_size)
        ]

    def sample_new_examples(self, steps, step_size, noise):
        n_new = np.random.binomial(self.pool_size, 0.05)
        new_rand_imgs = torch.rand(
            (n_new, self.num_channels, self.image_size, self.image_size), device=self.device
        ) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.pool_size - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        new_imgs = langevin_dynamics(self.model, inp_imgs, steps, step_size, noise)

        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[: self.max_size]
        return new_imgs


def train_ebm(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs=10,
    image_size=32,
    num_channels=3,
    steps=60,
    step_size=10.0,
    noise=0.005,
    alpha=0.1,
    checkpoint_dir="checkpoints_ebm",
):
    """Contrastive-divergence training loop for an EnergyModel, matching Practical 1's
    EBM.train_step: pushes energy down on real images and up on Langevin-sampled
    ("fake") images."""
    model.to(device)
    buffer = _EBMBuffer(model, device, image_size=image_size, num_channels=num_channels)

    avg_loss, avg_cdiv = 0.0, 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, running_cdiv = 0.0, 0.0

        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for real_imgs, _ in progress:
            real_imgs = real_imgs.to(device)
            real_imgs = torch.clamp(real_imgs + torch.randn_like(real_imgs) * noise, -1.0, 1.0)

            fake_imgs = buffer.sample_new_examples(steps=steps, step_size=step_size, noise=noise)

            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).detach()
            out_scores = model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

            cdiv_loss = real_out.mean() - fake_out.mean()
            reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()
            running_cdiv += cdiv_loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}", "cdiv": f"{cdiv_loss.item():.4f}"})

        avg_loss = running_loss / len(data_loader)
        avg_cdiv = running_cdiv / len(data_loader)

        save_checkpoint(model, optimizer, epoch + 1, avg_loss, 0.0, checkpoint_dir, filename="model_latest.pth")
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, cdiv={avg_cdiv:.4f}")

    save_checkpoint(model, optimizer, epochs, avg_loss, 0.0, checkpoint_dir=f"{checkpoint_dir}/best", filename="model.pth")

    return model


def train_diffusion(
    model,
    ema_model,
    data_loader,
    optimizer,
    schedule_fn,
    device="cpu",
    epochs=10,
    ema_decay=0.8,
    checkpoint_dir="checkpoints_diffusion",
):
    """Noise-prediction training loop for a diffusion UNet, matching Practical 2's
    DiffusionModel.train_step. ema_model is a shadow copy of model updated by an
    exponential moving average, used for generation at inference time."""
    model.to(device)
    ema_model.to(device)
    criterion = nn.L1Loss()

    avg_loss = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, _ in progress:
            images = images.to(device)
            noises = torch.randn_like(images)

            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=device)
            noise_rates, signal_rates = schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            pred_noises = model(noisy_images, noise_rates ** 2)
            loss = criterion(pred_noises, noises)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.copy_(ema_decay * ema_param + (1.0 - ema_decay) * param)

            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(data_loader)

        save_checkpoint(model, optimizer, epoch + 1, avg_loss, 0.0, checkpoint_dir, filename="unet_latest.pth")
        save_checkpoint(ema_model, optimizer, epoch + 1, avg_loss, 0.0, checkpoint_dir, filename="unet_ema_latest.pth")
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    save_checkpoint(model, optimizer, epochs, avg_loss, 0.0, checkpoint_dir=f"{checkpoint_dir}/best", filename="unet.pth")
    save_checkpoint(ema_model, optimizer, epochs, avg_loss, 0.0, checkpoint_dir=f"{checkpoint_dir}/best", filename="unet_ema.pth")

    return model, ema_model
