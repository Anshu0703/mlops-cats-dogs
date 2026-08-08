"""
Trains the SimpleCNN on Cats vs Dogs and logs everything to MLflow:
params, per-epoch metrics, loss curves, confusion matrix, and the model itself.

Usage:
    python src/train.py
    python src/train.py --epochs 10 --lr 0.001 --batch-size 32
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import SimpleCNN
from dataset import get_dataloaders

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  # (batch, 1) -> (batch,)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def main(epochs: int, lr: float, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("cats-vs-dogs-cnn")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "architecture": "SimpleCNN-3conv",
            "optimizer": "Adam",
            "device": str(device),
        })

        train_losses, val_losses = [], []

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(f"Epoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), train_losses, label="train_loss")
        ax.plot(range(1, epochs + 1), val_losses, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        loss_curve_path = MODELS_DIR / "loss_curve.png"
        fig.savefig(loss_curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(loss_curve_path))

        cm = confusion_matrix(test_labels, test_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["cat", "dog"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        cm_path = MODELS_DIR / "confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        model_path = MODELS_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.pytorch.log_model(model, name="pytorch_model")

        print(f"Model saved to {model_path}")
        print("MLflow run complete. Launch `mlflow ui` to view results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    main(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)