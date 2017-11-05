from functools import reduce

import torch
import torch.utils.data as data
from sklearn.datasets import fetch_mldata
import numpy as np


class LimitedMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset with digit limitation.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        digits (list, optional): List of digits to limit the dataset to.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, digits=[0], fraction=1.0):
        self.transform = transform
        self.target_transform = target_transform

        self.mnist = fetch_mldata("MNIST original", data_home=root)

        indices = np.arange(len(self.mnist.target))
        np.random.shuffle(indices)

        self.mnist.data = self.mnist.data[indices]
        self.mnist.target = self.mnist.target[indices]

        if train:
            self.mnist.data = self.mnist.data[:int(40000*fraction)]
            self.mnist.target = self.mnist.target[:int(40000*fraction)]
        else:
            self.mnist.data = self.mnist.data[40000:]
            self.mnist.target = self.mnist.target[40000:]

        # Filter digits
        filter = reduce(lambda x, y: x | y, [self.mnist.target == i for i in digits])
        self.mnist.data = self.mnist.data[filter]
        self.mnist.target = self.mnist.target[filter]


    def __getitem__(self, index):
        img, target = self.mnist.data[index], self.mnist.target[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.FloatTensor((img/255).astype(np.float32))
        target = torch.LongTensor(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.mnist.target)