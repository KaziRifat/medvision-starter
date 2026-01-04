import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data import DataConfig, build_dataloaders
from model import build_model
from utils import set_seed, get_device, save_checkpoint, accuracy_from_logits, write_json


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc=split_name, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="resnet18", choices=["simplecnn", "resnet18"])
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (resnet18)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    cfg = DataConfig(img_size=args.img_size, batch_size=args.batch_size)
    train_loader, val_loader, _, class_names = build_dataloaders(args.data_dir, cfg)

    model = build_model(args.model, num_classes=len(class_names), pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    best_val_acc = -1.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "class_names": class_names,
        "model": args.model
    }

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device, split_name="Val")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch}/{args.epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint("checkpoints/best.pt", model, optimizer, epoch, best_val_acc)

    write_json("outputs/training_history.json", history)
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print("Saved: checkpoints/best.pt and outputs/training_history.json")


if __name__ == "__main__":
    main()
