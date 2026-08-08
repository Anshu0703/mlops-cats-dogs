"""
Defines the CNN architecture for Cats vs Dogs binary classification.
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A small CNN for 224x224 RGB binary classification.
    3 conv blocks (conv -> relu -> maxpool) followed by a classifier head.
    Outputs a single logit (use sigmoid for probability, since this is binary).
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 112x112x16 -> 56x56x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 56x56x32 -> 28x28x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # single logit: >0 = dog, <0 = cat (after sigmoid: prob of "dog")
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: does a dummy batch flow through without shape errors?
    model = SimpleCNN()
    dummy_input = torch.randn(2, 3, 224, 224)  # batch of 2 images
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect torch.Size([2, 1])