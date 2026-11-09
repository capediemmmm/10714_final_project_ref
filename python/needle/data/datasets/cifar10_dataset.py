import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.train, self.p = train, p
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images, labels = [], []
        # Official Python batches store flattened images in R, G, B planes:
        # https://www.cs.toronto.edu/~kriz/cifar.html
        for filename in files:
            with open(os.path.join(base_folder, filename), "rb") as source:
                batch = pickle.load(source, encoding="bytes")
            images.append(batch[b"data"] if b"data" in batch else batch["data"])
            labels.extend(batch[b"labels"] if b"labels" in batch else batch["labels"])
        self.X = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).astype(np.float32)
        self.X /= 255.0
        self.y = np.asarray(labels, dtype=np.int64)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("CIFAR-10 image and label counts do not match")
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        images, labels = self.X[index], self.y[index]
        if self.transforms:
            # The shared image transforms take HWC, while CIFAR returns CHW.
            def transform(image):
                image = self.apply_transforms(image.transpose(1, 2, 0).copy())
                return image.transpose(2, 0, 1)

            if images.ndim == 3:
                images = transform(images)
            else:
                images = np.stack([transform(image) for image in images])
        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
