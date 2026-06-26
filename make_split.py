"""
make_split.py  —  evaluation-pipeline person, step 2.

Splits the raw downloaded images into the two folders the starter code needs:
    dataset/train/<class_name>/*.jpg        <- teammates train on this
    dataset/validation/<class_name>/*.jpg   <- evaluate.py scores on this

It reads the 20 official class names straight from your labels.py, so the
folder names always match what ImageNetSubset / evaluate.py look for.

WHERE ARE THE RAW IMAGES?
Pass --src pointing at the folder that directly contains the 20 class
subfolders (the thing you downloaded/unzipped). Examples:
    python make_split.py --src train_set
    python make_split.py --src dataset/train_set
The script copies (never deletes) into dataset/train and dataset/validation,
so it is safe to re-run.

    python make_split.py --src dataset/train_set --val-frac 0.2 --seed 42
"""
from __future__ import annotations
import argparse, random, shutil, sys
from pathlib import Path

from labels import HF_INDEX_TO_NAME   # the real 20 class names live here

CLASS_NAMES = sorted(HF_INDEX_TO_NAME.values())
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_source(src: Path) -> Path:
    """Locate the folder that actually holds the 20 class subfolders."""
    candidates = [src, *[d for d in src.iterdir() if d.is_dir()]] if src.is_dir() else []
    for cand in candidates:
        if cand.is_dir() and sum((cand / c).is_dir() for c in CLASS_NAMES) >= 15:
            return cand
    raise FileNotFoundError(
        f"Could not find the class folders under '{src}'. I expected to see "
        f"subfolders named like {CLASS_NAMES[:3]}... Point --src at the unzipped data."
    )


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing the 20 class subfolders")
    ap.add_argument("--out", default="dataset", help="where train/ and validation/ are created")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = find_source(Path(args.src))
    out = Path(args.out)
    rng = random.Random(args.seed)
    print(f"Reading raw images from: {source.resolve()}")

    totals = {"train": 0, "validation": 0}
    for class_name in CLASS_NAMES:
        imgs = list_images(source / class_name)
        if not imgs:
            print(f"  [warn] no images for class '{class_name}' — skipping")
            continue
        rng.shuffle(imgs)
        n_val = max(1, round(len(imgs) * args.val_frac))
        groups = {"validation": imgs[:n_val], "train": imgs[n_val:]}
        for split, files in groups.items():
            dst_dir = out / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst_dir / f.name)
            totals[split] += len(files)

    print(f"\nDone.")
    print(f"  dataset/train/        {totals['train']} images")
    print(f"  dataset/validation/   {totals['validation']} images")
    print(f"\nNow you can run:  python evaluate.py")


if __name__ == "__main__":
    main()