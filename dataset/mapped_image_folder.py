import os
import json
from typing import Dict, Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class JSONMappedImageFolder(ImageFolder):
    """
    A custom ImageFolder that overrides the default alphabetical class-to-index 
    mapping with a custom mapping provided by a JSON file.
    """
    def __init__(self, root: str, json_path: str, transform=None):
        # Load the JSON mapping first before calling super().__init__
        self.custom_mapping = self._parse_json_mapping(json_path)
        
        # super().__init__ will trigger find_classes internally
        super().__init__(root, transform=transform)

    def _parse_json_mapping(self, json_path: str) -> Dict[str, int]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the mapping is: { "folder_name": int_id }
        # If your JSON is { "0": "cat", "1": "dog" }, we invert it to { "cat": 0, "dog": 1 }
        inverted_mapping = {str(v): int(k) for k, v in data.items()}
        return inverted_mapping

    def find_classes(self, directory: str) -> Tuple[list, Dict[str, int]]:
        """
        Overrides PyTorch's default alphabetical directory scanning.
        """
        # 1. Get the class names (folder names) from our custom mapping
        classes = list(self.custom_mapping.keys())
        
        # 2. Validate that these folders actually exist in the directory
        for class_name in classes:
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class folder '{class_name}' defined in JSON was not found in {directory}")
                
        # 3. Return the list of classes and the explicit mapping dictionary
        return classes, self.custom_mapping