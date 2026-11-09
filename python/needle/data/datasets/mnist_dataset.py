from typing import List, Optional
from ..data_basic import Dataset
import gzip
import numpy as np
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as image_file:
            magic, count, rows, cols = struct.unpack(">IIII", image_file.read(16))
            if magic != 2051:
                raise ValueError("Invalid MNIST image magic number")
            self.images = np.frombuffer(image_file.read(), dtype=np.uint8)
            self.images = self.images.reshape(count, rows, cols, 1).astype(np.float32) / 255.0
        with gzip.open(label_filename, "rb") as label_file:
            magic, label_count = struct.unpack(">II", label_file.read(8))
            if magic != 2049:
                raise ValueError("Invalid MNIST label magic number")
            self.labels = np.frombuffer(label_file.read(), dtype=np.uint8)
        if label_count != count or len(self.labels) != count:
            raise ValueError("MNIST image and label counts do not match")
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images, labels = self.images[index], self.labels[index]
        if self.transforms:
            if images.ndim == 3:
                images = self.apply_transforms(images.copy())
            else:
                images = np.stack([self.apply_transforms(image.copy()) for image in images])
        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
