import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from data import DataConfig, build_dataloaders
from model import build_model
from utils import get_device, load_checkpoint, write_json


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_cm(cm, class_names, out_path):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet18", choices=["simplecnn", "resnet18"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = get_device()
    cfg = DataConfig(img_size=args.img_size, batch_size=args.batch_size)
    _, _, test_loader, class_names = build_dataloaders(args.data_dir, cfg)

    model = build_model(args.model, num_classes=len(class_names), pretrained=args.pretrained).to(device)
    load_checkpoint(args.checkpoint, model, device)

    preds, targets = predict_all(model, test_loader, device)

    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)

    os.makedirs("outputs", exist_ok=True)
    plot_cm(cm, class_names, "outputs/confusion_matrix.png")
    write_json("outputs/metrics.json", report)

    print("Saved: outputs/confusion_matrix.png and outputs/metrics.json")
    print("\nClassification report:\n")
    print(classification_report(targets, preds, target_names=class_names))


if __name__ == "__main__":
    main()
