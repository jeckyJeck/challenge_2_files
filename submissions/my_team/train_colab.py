from pathlib import Path
import sys

import joblib
import torch
import torch.nn as nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_process import ImageClassificationDataModule  # noqa: E402
from model import ModelArchitecture  # noqa: E402


OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PROGRESS_BAR_WIDTH = 30


def get_training_device():
    """Use Colab GPU when available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    use_cuda = device.type == "cuda"
    num_batches = len(loader)

    for batch_index, (x, y) in enumerate(loader, start=1):
        x = x.to(
            device=device,
            non_blocking=use_cuda,
            memory_format=torch.channels_last,
        )
        y = y.to(device=device, non_blocking=use_cuda)

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
    use_cuda = device.type == "cuda"
    num_batches = len(loader)

    for batch_index, (x, y) in enumerate(loader, start=1):
        x = x.to(
            device=device,
            non_blocking=use_cuda,
            memory_format=torch.channels_last,
        )
        y = y.to(device=device, non_blocking=use_cuda)

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
    Colab training pipeline.

    This script creates weights.joblib in submissions/my_team/.
    """
    device = get_training_device()
    print(f"Using device: {device}")

    data_module = ImageClassificationDataModule(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        test_split=0.0,
        num_workers=2,
        seed=42,
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
