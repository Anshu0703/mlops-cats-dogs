"""
PyTorch Dataset and DataLoader utilities for Cats vs Dogs.
Includes data augmentation for the training set.
"""
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Augmentation for training: random flips/rotation add variety so the model
# generalizes instead of memorizing exact pixel arrangements.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats, standard practice
])

# No augmentation for val/test — we want a stable, honest measure of performance.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CatsDogsDataset(Dataset):
    """Reads images from data/processed/<split>/<cat|dog>/*.jpg"""

    def __init__(self, split: str, transform=None):
        self.split_dir = DATA_DIR / split
        self.transform = transform
        self.samples = []  # list of (filepath, label) — 0=cat, 1=dog

        for label_name, label_idx in [("cat", 0), ("dog", 1)]:
            class_dir = self.split_dir / label_name
            for f in class_dir.glob("*.jpg"):
                self.samples.append((f, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(batch_size: int = 32):
    train_ds = CatsDogsDataset("train", transform=train_transform)
    val_ds = CatsDogsDataset("val", transform=eval_transform)
    test_ds = CatsDogsDataset("test", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader