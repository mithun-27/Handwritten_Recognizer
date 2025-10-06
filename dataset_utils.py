# dataset_utils.py
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torch
import os

# normalization: use MNIST statistics
MEAN = (0.1307,)
STD = (0.3081,)

def get_class_maps():
    idx_to_char = {i: str(i) for i in range(10)}
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        idx_to_char[10 + i] = ch
    char_to_idx = {v: k for k, v in idx_to_char.items()}
    return idx_to_char, char_to_idx

class RemapMNIST(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, label

class RemapEMNIST(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        new_label = 10 + (label - 1)
        return img, new_label

def transform_emnist(img):
    # EMNIST images are rotated and mirrored, this corrects them
    img = transforms.functional.rotate(img, -90)
    img = transforms.functional.hflip(img)  # Corrected from .transpose to .hflip
    return img

def load_datasets(root="./data", download=True):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    emnist_transform = transforms.Compose([
        transform_emnist,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    mnist_train = datasets.MNIST(root, train=True, download=download, transform=mnist_transform)
    mnist_test  = datasets.MNIST(root, train=False, download=download, transform=mnist_transform)

    emnist_train = datasets.EMNIST(root, split='letters', train=True, download=download, transform=emnist_transform)
    emnist_test  = datasets.EMNIST(root, split='letters', train=False, download=download, transform=emnist_transform)

    train = ConcatDataset([RemapMNIST(mnist_train), RemapEMNIST(emnist_train)])
    test  = ConcatDataset([RemapMNIST(mnist_test),  RemapEMNIST(emnist_test)])

    idx_to_char, char_to_idx = get_class_maps()
    return train, test, (idx_to_char, char_to_idx)
