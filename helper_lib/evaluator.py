import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.to(device)
    model.eval()

    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
