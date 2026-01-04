from dataclasses import dataclass
from typing import Tuple, List

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


@dataclass
class DataConfig:
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def build_dataloaders(data_dir: str, cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_ds = ImageFolder(f"{data_dir}/train", transform=build_transforms(cfg.img_size, train=True))
    val_ds   = ImageFolder(f"{data_dir}/val", transform=build_transforms(cfg.img_size, train=False))
    test_ds  = ImageFolder(f"{data_dir}/test", transform=build_transforms(cfg.img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names
