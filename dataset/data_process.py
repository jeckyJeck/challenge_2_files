import os
import json
from typing import Callable, Dict, Tuple, List
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

DATA_DIR = "train_set/train"
LABELS_JSON = "labels.json"

# --- Helper Classes inside your pipeline file ---

class JSONMappedImageFolder(ImageFolder):
    """Custom ImageFolder that maps classes based on JSON instead of alphabet."""
    def __init__(self, root: str, json_path: str, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.custom_mapping = {str(v): int(k) for k, v in data.items()}
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[list, Dict[str, int]]:
        classes = list(self.custom_mapping.keys())
        for class_name in classes:
            if not os.path.isdir(os.path.join(directory, class_name)):
                raise FileNotFoundError(f"Class folder '{class_name}' was not found in {directory}")
        return classes, self.custom_mapping

class SubsetWrapper(Dataset):
    """
    Wraps a PyTorch Subset to apply split-specific transforms 
    (e.g., augmentations for train, only resizing for val).
    """
    def __init__(self, subset: torch.utils.data.Subset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        original_index = self.subset.indices[index]
        
        # Explicitly tell your IDE that the underlying dataset is an ImageFolder
        original_dataset: ImageFolder = self.subset.dataset
        
        path, label = original_dataset.samples[original_index]
        img = original_dataset.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

    def __len__(self) -> int:
        return len(self.subset.indices)


# --- Main API Class ---

class ImageClassificationDataModule:
    def __init__(
        self,
        data_dir: str,
        class_mapping_json: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.class_mapping_json = class_mapping_json
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        
        # Load the class names from JSON
        with open(class_mapping_json, 'r', encoding='utf-8') as f:
            self.class_to_name = json.load(f)
            
        # Internal placeholders for the raw PyTorch subsets
        self._raw_train_subset = None
        self._raw_val_subset = None
        
        # Split the data into train/val raw structures
        self._prepare_datasets()

    def _get_transforms(self, augment: bool) -> transforms.Compose:
        """Centralized place for all image manipulations."""
        base_transforms = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if augment:
            # Heavy manipulations for training
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ]
            return transforms.Compose(augmentation_transforms + base_transforms)
            
        return transforms.Compose(base_transforms)

    def _prepare_datasets(self):
        """Initializes the dataset and performs the train/val split."""
        # We pass transform=None because SubsetWrapper will handle it lazily
        full_dataset = JSONMappedImageFolder(
            root=self.data_dir,
            json_path=self.class_mapping_json,
            transform=None
        )
        
        # Calculate split sizes
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        # Reproducible split using generator seed
        generator = torch.Generator().manual_seed(self.seed)
        self._raw_train_subset, self._raw_val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_to_name)

    def get_class_names(self) -> List[str]:
        return [self.class_to_name[str(i)] for i in range(self.num_classes)]

    # --- The requested API methods ---

    def get_train_loader(self, augment: bool = True) -> DataLoader:
        """Returns a DataLoader for training with optional augmentation."""
        train_transform = self._get_transforms(augment=augment)
        train_dataset = SubsetWrapper(self._raw_train_subset, transform=train_transform)
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Always shuffle training data
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available() # Speeds up transfer to GPU
        )

    def get_evaluation_loader(self) -> DataLoader:
        """Returns a DataLoader for validation/evaluation (No augmentation, no shuffle)."""
        val_transform = self._get_transforms(augment=False)
        val_dataset = SubsetWrapper(self._raw_val_subset, transform=val_transform)
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # Do not shuffle evaluation data
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
if __name__ == "__main__":
    # Example usage
    data_module = ImageClassificationDataModule(
        data_dir= DATA_DIR,
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