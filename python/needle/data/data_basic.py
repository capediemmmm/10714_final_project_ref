import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
        dtype="float32",
    ):

        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._batch_index = 0
        if not self.shuffle:
            self.ordering = [np.arange(i, min(i + batch_size, len(dataset)))
                             for i in range(0, len(dataset), batch_size)]

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
            self.ordering = [indices[i:i + self.batch_size]
                             for i in range(0, len(indices), self.batch_size)]
        self._batch_index = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self._batch_index >= len(self.ordering):
            raise StopIteration
        indices = self.ordering[self._batch_index]
        self._batch_index += 1
        batch = self.dataset[indices]
        if not isinstance(batch, tuple):
            batch = (batch,)
        return tuple(Tensor(array, device=self.device, dtype=self.dtype,
                            requires_grad=False) for array in batch)
        ### END YOUR SOLUTION

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

