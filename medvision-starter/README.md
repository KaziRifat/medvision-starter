# MedVision-Starter: Medical Image Classification (Beginner)

A clean beginner-friendly **PyTorch** repo for image classification (binary or multi-class).
Works with any dataset arranged in **ImageFolder** format:

```
data/
  train/
    class0/
    class1/
  val/
    class0/
    class1/
  test/
    class0/
    class1/
```

## 1) Setup (macOS)

```bash
cd medvision-starter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset

Put your images inside `data/train`, `data/val`, `data/test` as shown above.

If you currently have everything in one folder like:
```
data/all/class0/*.jpg
data/all/class1/*.jpg
```
you can split it automatically:

```bash
python3 scripts/split_folder_dataset.py --input_dir data/all --output_dir data --val 0.15 --test 0.15
```

## 3) Train

### Option A: Simple CNN (from scratch)
```bash
python3 src/train.py --data_dir data --model simplecnn --epochs 10 --img_size 224 --batch_size 32 --lr 1e-3
```

### Option B: Transfer Learning (ResNet-18)
```bash
python3 src/train.py --data_dir data --model resnet18 --pretrained --epochs 5 --img_size 224 --batch_size 32 --lr 1e-4
```

Outputs:
- `checkpoints/best.pt`
- `outputs/training_history.json`

## 4) Evaluate on test set

```bash
python3 src/evaluate.py --data_dir data --checkpoint checkpoints/best.pt --model resnet18
```

Outputs:
- `outputs/confusion_matrix.png`
- `outputs/metrics.json`

## Notes
- This repo is intentionally simple and beginner-friendly.
- You can extend it later with stronger augmentations, more models, or medical datasets.
