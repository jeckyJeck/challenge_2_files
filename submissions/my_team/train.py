import json
import os
import shutil
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import joblib
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        dataset_dir = candidate / "dataset"
        if (dataset_dir / "train_set").exists() and (
            dataset_dir / "labels.json"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find project root. Expected dataset/train_set and "
        "dataset/labels.json as described in section 4.1."
    )


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ModelArchitecture  # noqa: E402

OUTPUT = SCRIPT_DIR / "weights.joblib"

DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = DATASET_DIR / "train_set"
LABELS_JSON = DATASET_DIR / "labels.json"
SPLITS_DIR = DATASET_DIR
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
TEST_SPLIT = 0.0
SPLIT_SEED = 42
NUM_WORKERS = 0
ACCELERATOR_DEVICE_TYPES = {"cuda", "xpu"}
PROGRESS_BAR_WIDTH = 30


class JSONMappedImageFolder(ImageFolder):
    """ImageFolder that preserves the class-to-index mapping from labels.json."""

    def __init__(self, root: str, json_path: str, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.custom_mapping = {
            str(class_name): int(label) for label, class_name in data.items()
        }
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[list, Dict[str, int]]:
        classes = [
            class_name
            for class_name, _ in sorted(
                self.custom_mapping.items(), key=lambda item: item[1]
            )
        ]
        for class_name in classes:
            if not os.path.isdir(os.path.join(directory, class_name)):
                raise FileNotFoundError(
                    f"Class folder '{class_name}' was not found in {directory}"
                )
        return classes, self.custom_mapping


class SubsetWrapper(Dataset):
    """
    Wraps a PyTorch Subset to apply split-specific transforms
    (e.g., augmentations for train, only resizing for val).
    """

    def __init__(self, subset: Subset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        original_index = self.subset.indices[index]
        original_dataset: ImageFolder = self.subset.dataset

        path, label = original_dataset.samples[original_index]
        img = original_dataset.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self) -> int:
        return len(self.subset.indices)


class ImageClassificationDataModule:
    """
    Self-contained data pipeline based on the original dataset/data_process.py.
    The only behavioral change is that deterministic splits are materialized on disk.
    """

    def __init__(
        self,
        data_dir: str = str(DATA_DIR),
        class_mapping_json: str = str(LABELS_JSON),
        split_root: str = str(SPLITS_DIR),
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.0,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.class_mapping_json = Path(class_mapping_json)
        self.split_root = Path(split_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.seed = seed

        if not (0 <= self.test_split < 1 and 0 <= self.val_split < 1):
            raise ValueError("test_split and val_split must be between 0 and 1.")
        if self.val_split + self.test_split >= 1:
            raise ValueError("val_split + test_split must be less than 1.")

        with open(self.class_mapping_json, "r", encoding="utf-8") as f:
            self.class_to_name = json.load(f)

        self._raw_train_subset = None
        self._raw_val_subset = None
        self._raw_test_subset = None
        self._train_augmentations = self._get_default_train_augmentations()

        self._prepare_datasets()

    def _get_base_transform_steps(self) -> List[Callable]:
        return [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    def _get_default_train_augmentations(self) -> List[Callable]:
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]

    def set_transforms(self, list_of_transforms: Sequence[Callable]) -> None:
        """Replace train-time augmentations. These run before ToTensor/Normalize."""
        self._train_augmentations = list(list_of_transforms)

    def _get_transforms(self, augment: bool) -> transforms.Compose:
        augmentation_steps = self._train_augmentations if augment else []
        return transforms.Compose(
            [*augmentation_steps, *self._get_base_transform_steps()]
        )

    def _copy_or_link(self, source: Path, target: Path) -> None:
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _materialize_split(self, split_name: str, subset: Subset) -> None:
        split_dir = self.split_root / split_name
        for index in subset.indices:
            source_path, _ = subset.dataset.samples[index]
            source_path = Path(source_path)
            target_path = split_dir / source_path.parent.name / source_path.name
            self._copy_or_link(source_path, target_path)

    def _prepare_datasets(self) -> None:
        """Initializes the dataset, performs deterministic stratified splits, and saves them."""
        full_dataset = JSONMappedImageFolder(
            root=str(self.data_dir),
            json_path=str(self.class_mapping_json),
            transform=None,
        )

        generator = torch.Generator().manual_seed(self.seed)
        indices_by_label = defaultdict(list)

        for sample_index, (_, label) in enumerate(full_dataset.samples):
            indices_by_label[label].append(sample_index)

        train_indices = []
        val_indices = []
        test_indices = []

        for label in sorted(indices_by_label):
            class_indices = indices_by_label[label]
            permutation = torch.randperm(
                len(class_indices), generator=generator
            ).tolist()
            shuffled_indices = [class_indices[i] for i in permutation]

            test_size = int(len(shuffled_indices) * self.test_split)
            val_size = int(len(shuffled_indices) * self.val_split)

            test_indices.extend(shuffled_indices[:test_size])
            val_indices.extend(shuffled_indices[test_size : test_size + val_size])
            train_indices.extend(shuffled_indices[test_size + val_size :])

        self._raw_train_subset = Subset(full_dataset, train_indices)
        self._raw_val_subset = Subset(full_dataset, val_indices)
        self._raw_test_subset = Subset(full_dataset, test_indices)

        self._materialize_split("train", self._raw_train_subset)
        self._materialize_split("validation", self._raw_val_subset)
        if len(test_indices) > 0:
            self._materialize_split("test", self._raw_test_subset)

    @property
    def num_classes(self) -> int:
        return len(self.class_to_name)

    def get_class_names(self) -> List[str]:
        return [self.class_to_name[str(i)] for i in range(self.num_classes)]

    def get_train_loader(self, augment: bool = True) -> DataLoader:
        """Returns a DataLoader for training with optional augmentation."""
        train_transform = self._get_transforms(augment=augment)
        train_dataset = SubsetWrapper(self._raw_train_subset, transform=train_transform)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_evaluation_loader(self) -> DataLoader:
        """Returns a DataLoader for validation/evaluation (No augmentation, no shuffle)."""
        val_transform = self._get_transforms(augment=False)
        val_dataset = SubsetWrapper(self._raw_val_subset, transform=val_transform)

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_val_loader(self) -> DataLoader:
        """Alias for callers that prefer explicit validation naming."""
        return self.get_evaluation_loader()

    def get_test_loader(self) -> DataLoader:
        """Returns a DataLoader for testing (No augmentation, no shuffle)."""
        if self._raw_test_subset is None or len(self._raw_test_subset) == 0:
            raise ValueError(
                "Test set not available. Initialize with test_split > 0 to enable test set."
            )

        test_transform = self._get_transforms(augment=False)
        test_dataset = SubsetWrapper(self._raw_test_subset, transform=test_transform)

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def load_ipex():
    try:
        return import_module("intel_extension_for_pytorch")
    except ImportError:
        return None


def get_training_device():
    """
    Prefer Intel Arc through PyTorch's XPU backend, then CUDA, then CPU.
    Importing IPEX first lets older Intel PyTorch builds register XPU support.
    """
    ipex = load_ipex()

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), ipex

    if torch.cuda.is_available():
        return torch.device("cuda"), ipex

    return torch.device("cpu"), ipex


def print_progress(phase, epoch, batch_index, num_batches, loss, accuracy):
    filled = int(PROGRESS_BAR_WIDTH * batch_index / num_batches)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(
        f"\rEpoch {epoch:02d}/{EPOCHS} {phase:<5} "
        f"[{bar}] {batch_index:>4}/{num_batches} "
        f"loss {loss:.4f} acc {accuracy:.4f}",
        end="",
        flush=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_accelerator = device.type in ACCELERATOR_DEVICE_TYPES
    num_batches = len(loader)

    for batch_index, (x, y) in enumerate(loader, start=1):
        x = x.to(
            device=device,
            non_blocking=use_accelerator,
            memory_format=torch.channels_last,
        )
        y = y.to(device=device, non_blocking=use_accelerator)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

        print_progress(
            "train",
            epoch,
            batch_index,
            num_batches,
            total_loss / total,
            correct / total,
        )

    print()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    use_accelerator = device.type in ACCELERATOR_DEVICE_TYPES
    num_batches = len(loader)

    for batch_index, (x, y) in enumerate(loader, start=1):
        x = x.to(
            device=device,
            non_blocking=use_accelerator,
            memory_format=torch.channels_last,
        )
        y = y.to(device=device, non_blocking=use_accelerator)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

        print_progress(
            "val",
            epoch,
            batch_index,
            num_batches,
            total_loss / total,
            correct / total,
        )

    print()
    return total_loss / total, correct / total


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    device, ipex = get_training_device()
    print(f"Using device: {device}")
    if device.type == "xpu":
        print("Intel Arc GPU acceleration enabled through PyTorch XPU.")
    elif hasattr(torch, "xpu") and not torch.xpu.is_available():
        print(
            "PyTorch XPU is unavailable; install an XPU-enabled PyTorch/IPEX "
            "build to train on Intel Arc."
        )
    elif ipex is None:
        print("IPEX is not installed; falling back to the best native PyTorch device.")

    data_module = ImageClassificationDataModule(
        data_dir=str(DATA_DIR),
        class_mapping_json=str(LABELS_JSON),
        split_root=str(SPLITS_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        num_workers=NUM_WORKERS,
        seed=SPLIT_SEED,
    )

    train_loader = data_module.get_train_loader(augment=True)
    val_loader = data_module.get_val_loader()

    model = ModelArchitecture(num_classes=data_module.num_classes).to(
        device=device,
        memory_format=torch.channels_last,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    if ipex is not None and device.type == "xpu":
        model, optimizer = ipex.optimize(
            model,
            optimizer=optimizer,
            dtype=torch.float32,
        )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, val acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    joblib.dump(best_state, OUTPUT)
    print(f"Saved best model to {OUTPUT}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
