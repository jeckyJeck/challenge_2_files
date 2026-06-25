import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASET_DIR = Path(__file__).resolve().parent
DATA_DIR = DATASET_DIR / "train_set"
LABELS_JSON = DATASET_DIR / "labels.json"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class JSONMappedImageFolder(ImageFolder):
    """Custom ImageFolder that maps classes based on JSON instead of alphabet."""

    def __init__(self, root: str, json_path: str, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.custom_mapping = {str(class_name): int(label) for label, class_name in data.items()}
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[list, Dict[str, int]]:
        classes = [
            class_name
            for class_name, _ in sorted(self.custom_mapping.items(), key=lambda item: item[1])
        ]
        for class_name in classes:
            if not os.path.isdir(os.path.join(directory, class_name)):
                raise FileNotFoundError(f"Class folder '{class_name}' was not found in {directory}")
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
    """Prepares virtual train/validation/test splits for CNN training."""

    def __init__(
        self,
        data_dir: str = str(DATA_DIR),
        class_mapping_json: str = str(LABELS_JSON),
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.0,
        num_workers: int = 4,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.class_mapping_json = Path(class_mapping_json)
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
        return transforms.Compose([*augmentation_steps, *self._get_base_transform_steps()])

    def _prepare_datasets(self) -> None:
        """Initializes the dataset and performs deterministic virtual stratified splits."""
        full_dataset = JSONMappedImageFolder(
            root=self.data_dir,
            json_path=self.class_mapping_json,
            transform=None
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
            permutation = torch.randperm(len(class_indices), generator=generator).tolist()
            shuffled_indices = [class_indices[i] for i in permutation]

            test_size = int(len(shuffled_indices) * self.test_split)
            val_size = int(len(shuffled_indices) * self.val_split)

            test_indices.extend(shuffled_indices[:test_size])
            val_indices.extend(shuffled_indices[test_size:test_size + val_size])
            train_indices.extend(shuffled_indices[test_size + val_size:])

        self._raw_train_subset = Subset(full_dataset, train_indices)
        self._raw_val_subset = Subset(full_dataset, val_indices)
        self._raw_test_subset = Subset(full_dataset, test_indices)

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
            pin_memory=torch.cuda.is_available()
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
            pin_memory=torch.cuda.is_available()
        )

    def get_val_loader(self) -> DataLoader:
        """Alias for callers that prefer explicit validation naming."""
        return self.get_evaluation_loader()

    def get_test_loader(self) -> DataLoader:
        """Returns a DataLoader for testing (No augmentation, no shuffle).

        Only available when test_split > 0. Raises ValueError otherwise.
        """
        if self._raw_test_subset is None or len(self._raw_test_subset) == 0:
            raise ValueError("Test set not available. Initialize with test_split > 0 to enable test set.")

        test_transform = self._get_transforms(augment=False)
        test_dataset = SubsetWrapper(self._raw_test_subset, transform=test_transform)

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

if __name__ == "__main__":
    # Example usage with 2-way split (default)
    data_module = ImageClassificationDataModule(
        data_dir=DATA_DIR,
        class_mapping_json=LABELS_JSON,
        image_size=(224, 224),
        batch_size=32,
        val_split=0.2,
        num_workers=4,
        seed=42
    )

    train_loader = data_module.get_train_loader(augment=True)
    val_loader = data_module.get_evaluation_loader()

    print(f"Number of classes: {data_module.num_classes}")
    print(f"Class names: {data_module.get_class_names()}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Example usage with 3-way split (train/val/test)
    print("\n--- 3-way split example ---")
    data_module_3way = ImageClassificationDataModule(
        data_dir=DATA_DIR,
        class_mapping_json=LABELS_JSON,
        image_size=(224, 224),
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        num_workers=4,
        seed=42
    )

    train_loader_3way = data_module_3way.get_train_loader(augment=True)
    val_loader_3way = data_module_3way.get_evaluation_loader()
    test_loader_3way = data_module_3way.get_test_loader()

    print(f"Number of training batches: {len(train_loader_3way)}")
    print(f"Number of validation batches: {len(val_loader_3way)}")
    print(f"Number of test batches: {len(test_loader_3way)}")