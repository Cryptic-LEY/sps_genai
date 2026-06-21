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
