import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def copy_files(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out_dir / f.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with class subfolders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder with train/val/test")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError("input_dir must contain class subfolders");

    for cdir in class_dirs:
        images = [p for p in cdir.iterdir() if p.is_file() and is_image(p)]
        random.shuffle(images)

        n = len(images)
        n_test = int(n * args.test)
        n_val = int(n * args.val)

        test_files = images[:n_test]
        val_files = images[n_test:n_test + n_val]
        train_files = images[n_test + n_val:]

        copy_files(train_files, output_dir / "train" / cdir.name)
        copy_files(val_files, output_dir / "val" / cdir.name)
        copy_files(test_files, output_dir / "test" / cdir.name)

        print(f"{cdir.name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


    print("Done.")


if __name__ == "__main__":
    main()
