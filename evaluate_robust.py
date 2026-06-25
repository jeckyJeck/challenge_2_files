"""
evaluate_robust.py  —  the evaluation-pipeline person's main tool.

A companion to the provided evaluate.py. Where evaluate.py only measures
CLEAN accuracy, this also measures ROBUST accuracy: it re-runs the SAME
validation images after applying label-preserving corruptions (background /
lighting / colour changes), then reports a grader-style combined score:

        combined = 0.5 * clean_accuracy + 0.5 * mean_accuracy_on_unseen_corruptions

It loads a team's model exactly the way the grader does (through predict.py's
Model class), so what you measure here is faithful to the real evaluation.

Run from the PROJECT ROOT (the folder with labels.py and evaluate.py):

    python evaluate_robust.py                       # scores submissions/my_team
    python evaluate_robust.py --team dummy_baseline  # scores any team folder
"""
from __future__ import annotations
import argparse, importlib.util, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from base_model import ImageNetSubset
from robust_transforms import (
    ALL_FAMILIES, TRAIN_FAMILIES, HELDOUT_FAMILIES, identity,
)

# --- preprocessing: identical to the provided evaluate.py ---------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DATA_ROOT = Path("dataset")
N_CLASSES = 20


def make_transform(corruption):
    """PIL corruption -> the model's standard preprocessing -> tensor."""
    return transforms.Compose([
        transforms.Lambda(lambda im: corruption(im)),   # label-preserving change
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_team_model(team_dir: Path):
    """Load a team's Model through predict.py, exactly like evaluate.py does."""
    predict_path = team_dir / "predict.py"
    weights_path = team_dir / "weights.joblib"
    for p in (predict_path, weights_path, team_dir / "model.py"):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p.name} in {team_dir}")

    sys.path.insert(0, str(team_dir))
    sys.modules.pop("model", None)
    try:
        spec = importlib.util.spec_from_file_location(f"{team_dir.name}_predict", predict_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model()
        model.load(str(weights_path))
    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)
    return model


@torch.no_grad()
def accuracy(model, loader):
    correct = total = 0
    per_cls_c = defaultdict(int)
    per_cls_t = defaultdict(int)
    for x, y in loader:
        preds = model.predict(x)
        correct += (preds == y).sum().item()
        total += y.size(0)
        for yi, pi in zip(y.tolist(), preds.tolist()):
            per_cls_t[yi] += 1
            per_cls_c[yi] += int(pi == yi)
    acc = correct / max(total, 1)
    per_cls = {c: per_cls_c[c] / max(per_cls_t[c], 1) for c in range(N_CLASSES)}
    return acc, per_cls


def loader_for(corruption, batch_size, workers):
    ds = ImageNetSubset(DATA_ROOT, split="validation", transform=make_transform(corruption))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="my_team", help="folder name inside submissions/")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)   # 0 is safest on Windows
    args = ap.parse_args()

    team_dir = Path("submissions") / args.team
    print(f"Loading model from {team_dir} ...")
    model = load_team_model(team_dir)

    print("Scoring CLEAN validation ...")
    clean_acc, clean_per_cls = accuracy(model, loader_for(identity, args.batch_size, args.workers))

    per_corruption = {}
    for name, fn in ALL_FAMILIES.items():
        print(f"Scoring corruption: {name} ...")
        acc, _ = accuracy(model, loader_for(fn, args.batch_size, args.workers))
        per_corruption[name] = acc

    import statistics as st
    mean_train = st.mean(per_corruption[n] for n in TRAIN_FAMILIES)
    mean_held  = st.mean(per_corruption[n] for n in HELDOUT_FAMILIES)
    combined = 0.5 * clean_acc + 0.5 * mean_held   # held-out = honest OOD proxy

    print("\n================= ROBUSTNESS REPORT =================")
    print(f"team:                               {args.team}")
    print(f"clean (in-domain) accuracy          : {clean_acc:.4f}")
    print(f"mean acc, TRAIN-style corruptions   : {mean_train:.4f}")
    print(f"mean acc, HELD-OUT (unseen) corrupts : {mean_held:.4f}")
    print(f"--> COMBINED grader-style score      : {combined:.4f}")
    print("\nper-corruption accuracy (worst first):")
    for name, acc in sorted(per_corruption.items(), key=lambda kv: kv[1]):
        tag = "HELDOUT" if name in HELDOUT_FAMILIES else "train  "
        print(f"  {name:<16} [{tag}]  {acc:.4f}   (drop {clean_acc-acc:+.4f})")
    weak = sorted(clean_per_cls.items(), key=lambda kv: kv[1])[:5]
    print("\n5 weakest classes (clean):")
    for c, a in weak:
        print(f"  class {c:>2}: {a:.4f}")
    print("====================================================\n")


if __name__ == "__main__":
    main()