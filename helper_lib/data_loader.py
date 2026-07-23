from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_data_loader(data_dir="./data", batch_size=32, train=True, image_size=64, normalize=False):
    transform_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        # Rescale to [-1, 1], matching the EBM/diffusion buffers and outputs
        transform_steps.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform_steps)
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


def get_mnist_loader(data_dir="./data", batch_size=64, train=True):
    # Normalize to [-1, 1] to match the Generator's Tanh output range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(
        root=data_dir, train=train, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
