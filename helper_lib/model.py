import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import UNet


class FCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """CNN matching the Assignment 2 specification for a 64x64x3 input:
    Conv(16,3x3,s1,p1) -> ReLU -> MaxPool(2x2,s2)
    Conv(32,3x3,s1,p1) -> ReLU -> MaxPool(2x2,s2)
    Flatten -> FC(100) -> ReLU -> FC(num_classes)
    """

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Energy model matching the Practical 1 (Energy Based Methods) architecture,
    adapted for 3-channel 32x32 CIFAR-10 input instead of 1-channel MNIST.
    """

    def __init__(self, num_channels=3):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


class Generator(nn.Module):
    """Generator matching the Assignment 3 spec for 28x28x1 MNIST output:
    Linear(z_dim -> 7*7*128) -> reshape
    ConvTranspose2d(128,64,k4,s2,p1) -> BatchNorm2d -> ReLU      (7x7 -> 14x14)
    ConvTranspose2d(64,1,k4,s2,p1) -> Tanh                        (14x14 -> 28x28)
    """

    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = torch.tanh(self.deconv2(x))
        return x


class Discriminator(nn.Module):
    """Discriminator matching the Assignment 3 spec for 28x28x1 MNIST input:
    Conv2d(1,64,k4,s2,p1) -> LeakyReLU(0.2)                       (28x28 -> 14x14)
    Conv2d(64,128,k4,s2,p1) -> BatchNorm2d -> LeakyReLU(0.2)       (14x14 -> 7x7)
    Flatten -> Linear(128*7*7, 1)
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


_MODELS = {
    "FCNN": FCNN,
    "CNN": CNN,
    "EnhancedCNN": EnhancedCNN,
}


def get_model(model_name, num_classes=10, z_dim=100, num_channels=3, image_size=64):
    if model_name == "GAN":
        return Generator(z_dim=z_dim), Discriminator()
    if model_name == "EBM":
        return EnergyModel(num_channels=num_channels)
    if model_name == "Diffusion":
        return UNet(image_size=image_size, num_channels=num_channels)
    if model_name not in _MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose from {list(_MODELS) + ['GAN', 'EBM', 'Diffusion']}"
        )
    return _MODELS[model_name](num_classes=num_classes)
