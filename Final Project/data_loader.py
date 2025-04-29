import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ImageLabelDataset(Dataset):
    def __init__(self, image_file, label_file, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.image_data = self._load_images(image_file)
        self.label_data = self._load_labels(label_file)
        assert len(self.image_data) == len(self.label_data), "Number of images and labels must match!"

    def _load_images(self, file_path):
        # TODO: Implement file parsing based on the actual data format
        # This is a placeholder - you NEED to adapt this based on how
        # images are stored in the files (e.g., one image per line,
        # fixed width characters, etc.)
        print(f"Loading images from: {file_path}")
        images = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            img = []
            for i, line in enumerate(lines):
                # Assuming each line is a row of the image
                # Modify this part extensively based on actual format!
                # Example for fixed width representation:
                line = line.rstrip('\n')  # Remove trailing newline IMPORTANTLY
                if len(line) == self.img_width:  # Check if line looks like image row data
                    img.extend([1.0 if c in ['#', '+'] else 0.0 for c in line])  # Example: Binary features
                # Detect image boundaries (e.g., after height lines)
                if len(img) == self.img_height * self.img_width:
                    images.append(np.array(img, dtype=np.float32))
                    img = []  # Reset for next image
                elif len(line) != self.img_width and len(img) > 0:
                    print(f"Warning: Potential parsing issue at line {i + 1} in {file_path}. Img buffer size: {len(img)}")
                    # Decide how to handle potentially incomplete images or parsing errors
                    img = []  # Discard potentially corrupt buffer

        print(f"Loaded {len(images)} images.")
        if not images:
            raise ValueError(f"No images loaded from {file_path}. Check parsing logic and file format.")
        return torch.tensor(np.array(images))  # Shape: (num_images, height * width)

    def _load_labels(self, file_path):
        # TODO: Implement label file parsing
        print(f"Loading labels from: {file_path}")
        with open(file_path, 'r') as f:
            # Assuming one label per line, convert to integer
            labels = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(labels)} labels.")
        return torch.tensor(labels, dtype=torch.long)  # Use torch.long for CrossEntropyLoss

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.label_data[idx]
        return image, label


def load_digit_dataset(image_file, label_file, img_height, img_width):
    """
    Loads the digit dataset using the ImageLabelDataset class.

    Args:
        image_file (str): Path to the image file.
        label_file (str): Path to the label file.
        img_height (int): Height of the digit images.
        img_width (int): Width of the digit images.

    Returns:
        ImageLabelDataset: A dataset object for the digit data.
    """
    return ImageLabelDataset(image_file, label_file, img_height, img_width)