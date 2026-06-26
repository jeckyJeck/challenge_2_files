import sys
from importlib import import_module
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# from dataset.data_process import ImageClassificationDataModule  # noqa: E402
from model import ModelArchitecture  # noqa: E402

OUTPUT = Path("weights.joblib")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ACCELERATOR_DEVICE_TYPES = {"cuda", "xpu"}
PROGRESS_BAR_WIDTH = 30


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


def get_train_loader():
    train_dir = PROJECT_ROOT / "dataset" / "train"
    train_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ImageFolder(train_dir, train_transforms)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def get_val_loader():
    val_dir = PROJECT_ROOT / "dataset" / "validation"
    val_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = ImageFolder(val_dir, val_transforms)
    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


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

    # data_module = ImageClassificationDataModule(
    #     image_size=IMAGE_SIZE,
    #     batch_size=BATCH_SIZE,
    #     val_split=0.2,
    #     test_split=0.0,
    #     num_workers=0,
    #     seed=42,
    # )

    # train_loader = data_module.get_train_loader(augment=True)
    # val_loader = data_module.get_val_loader()

    train_loader = get_train_loader()
    val_loader = get_val_loader()

    model = ModelArchitecture(num_classes=20).to(
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
