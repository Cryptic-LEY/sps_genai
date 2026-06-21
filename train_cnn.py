import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import get_device

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_data_loader(batch_size=32, train=True)
    test_loader = get_data_loader(batch_size=32, train=False)

    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device=device,
        epochs=5,
        checkpoint_dir="checkpoints",
    )

    avg_loss, accuracy = evaluate_model(trained_model, test_loader, criterion, device=device)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
