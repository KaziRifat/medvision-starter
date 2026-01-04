import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Small CNN for beginners (good for quick baselines)."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _resnet18(pretrained: bool):
    """Compatibility helper for different torchvision versions."""
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    except Exception:
        return models.resnet18(pretrained=pretrained)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower().strip()

    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)

    if name == "resnet18":
        m = _resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}. Use 'simplecnn' or 'resnet18'.")
