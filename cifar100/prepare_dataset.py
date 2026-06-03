#! /usr/bin/env python3   
import sys, os
import pickle
import numpy as np
import pandas as pd 
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

from kan_utils.dataset import group

# ----------------------------------------------------------------------
__dataset_dir = Path(__file__).parent / "dataset"
__train_file = "cifar100_train.pkl"
__test_file = "cifar100_test.pkl"
__labels_file = "labels.json"
__stats_file = "statistics.json"
__train_path = __dataset_dir / __train_file
__test_path = __dataset_dir / __test_file
__labels_path = __dataset_dir / __labels_file
__stats_path = __dataset_dir / __stats_file

# ----------------------------------------------------------------------

def build_dataset(force: bool = False) -> None:
    """
    Download CIFAR-100 (via torchvision) and save as pickle files.
    If force is True, existing files are overwritten.
    """
    if not force and __train_path.exists() and __test_path.exists():
        # print("Dataset already exists. Use force=True to rebuild.")
        return

    __dataset_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading CIFAR-100 dataset...")

    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError("torchvision is required to download CIFAR-100. Install with: pip install torchvision")

    # Download train and test sets
    train_ds = datasets.CIFAR100(root=str(__dataset_dir), train=True, download=True)
    test_ds = datasets.CIFAR100(root=str(__dataset_dir), train=False, download=True)

    # Convert to numpy arrays
    train_data = train_ds.data                     # (50000, 32, 32, 3)
    train_labels = np.array(train_ds.targets)      # (50000,)
    test_data = test_ds.data                       # (10000, 32, 32, 3)
    test_labels = np.array(test_ds.targets)        # (10000,)
    class_names = train_ds.classes                 # list of 100 strings

    # Save as pickle files
    with open(__train_path, "wb") as f:
        pickle.dump({"data": train_data, "labels": train_labels, "classes": class_names}, f)
    with open(__test_path, "wb") as f:
        pickle.dump({"data": test_data, "labels": test_labels, "classes": class_names}, f)

    print(f"Train: {train_data.shape}, Test: {test_data.shape}, Classes: {len(class_names)}")


def get_class_names() -> List[str]:
    """Retrieve the list of class names from the saved pickle or from torchvision."""
    # First try from the train pickle (fastest)
    if __train_path.exists():
        with open(__train_path, "rb") as f:
            data = pickle.load(f)
        if "classes" in data:
            return data["classes"]

    # Fallback to torchvision (without downloading)
    try:
        from torchvision import datasets
        ds = datasets.CIFAR100(root=str(__dataset_dir), train=True, download=False)
        return ds.classes
    except Exception as e:
        raise RuntimeError("Unable to obtain class names. Is the dataset downloaded?") from e


def create_labels(force: bool = False, save: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Create and optionally save a JSON mapping from class name to integer label.
    Returns a dictionary: {'class': {'class_name': idx, ...}}
    """
    if not save:
        # Just return the mapping without saving
        class_names = get_class_names()
        return {"class": {name: idx for idx, name in enumerate(class_names)}}

    if force or not __labels_path.exists():
        class_names = get_class_names()
        label_dict = {"class": {name: idx for idx, name in enumerate(class_names)}}
        __labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(__labels_path, "w") as f:
            json.dump(label_dict, f, indent=4)
        print(f"Labels saved to {__labels_path}")

    # Load and return the saved labels
    with open(__labels_path, "r") as f:
        return json.load(f)


def calculate_statistics(force: bool = False, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per‑channel mean and std on the training set (pixel values in [0,1]).
    Returns (mean, std) as numpy arrays of shape (3,).
    """
    if not save:
        # Compute on the fly
        return _compute_stats_from_train()

    if force or not __stats_path.exists():
        mean, std = _compute_stats_from_train()
        stats = {"mean": mean.tolist(), "std": std.tolist()}
        with open(__stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to {__stats_path}")
    else:
        with open(__stats_path, "r") as f:
            stats = json.load(f)
        mean, std = np.array(stats["mean"]), np.array(stats["std"])

    return mean, std


def _compute_stats_from_train() -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper: load train data and compute mean/std."""
    if not __train_path.exists():
        raise FileNotFoundError(f"Training data not found at {__train_path}. Run build_dataset() first.")
    with open(__train_path, "rb") as f:
        data_dict = pickle.load(f)
    data = data_dict["data"].astype(np.float32) / 255.0   # (N,32,32,3)
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std


def get_dataset(
    subset: Literal["train_val", "test"],
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple]:
    """
    Load CIFAR-100 data for the requested subset.

    :param subset: Options are 'train_val' or 'test'.
    :type subset: str

    :rtype: (data, labels) as requested.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if subset == "test":
        file_path = __test_path
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
        data = data_dict["data"]
        labels = data_dict["labels"]
    else:
        # Load train set and split
        with open(__train_path, "rb") as f:
            data_dict = pickle.load(f)
        data = data_dict["data"]
        labels = data_dict["labels"]

    return data, labels

def make_groups(df) :
    return group(
        df, 
        labels = ['Label']
    )
    
def __save_groups(groups):
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    with open(groups_path, 'w') as fw:
        json.dump(groups, fw, indent=2)
    
def __nested_pop(groups, keys):
    if len(keys) > 1 and keys[0] in groups.keys():
        __nested_pop(groups[keys[0]], keys[1:])
        if len(groups[keys[0]]) == 0:
            groups.pop(keys[0])
    elif len(keys) == 1 and keys[0] in groups.keys():
        groups.pop(keys[0])
    
def exclude_groups(groups, exclude : list[list[str]]=None):
    if exclude is not None:
        for keys in exclude:
            __nested_pop(groups, keys)
    return groups
    
def get_groups(regenerate = False, exclude=None):
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    
    if os.path.exists(groups_path) and not regenerate:
        with open(groups_path, 'r') as fr:
            groups = json.load(fr)
    else :
        _, labels = get_dataset('train_val')
        groups = make_groups(pd.DataFrame(data=labels, columns=['Label']))
        # print(groups)
        __save_groups(groups)
    return exclude_groups(groups, exclude)

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardise image data using pre‑computed per‑channel mean and std.
    Handles both flattened (N,3072) and image‑shaped inputs.
    """
    mean, std = calculate_statistics(save=False)   # load or compute

    # If data is still in [0,255] uint8, scale to [0,1] first
    if data.dtype == np.uint8:
        data = data.astype(np.float32) / 255.0

    if data.ndim == 2 and data.shape[1] == 3072:
        # Flattened: (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
        batch = data.shape[0]
        img = data.reshape(batch, 3, 32, 32).transpose(0, 2, 3, 1)
        img = (img - mean) / std
        # Flatten back to original format
        data = img.transpose(0, 3, 1, 2).reshape(batch, -1)
    elif data.ndim == 4:
        # Assume (batch, height, width, channels) or (batch, channels, height, width)
        if data.shape[1] == 3:   # (N, C, H, W)
            data = data.transpose(0, 2, 3, 1)   # -> (N, H, W, C)
        data = (data - mean) / std
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    return data


def get_dataset_paths(
    subsets: Union[Literal["train", "val", "test"], List[Literal["train", "val", "test"]]],
) -> Union[Path, List[Path], Dict[str, Path]]:
    """
    Return file path(s) for the requested subsets.
    For 'train' and 'val' the same train pickle is returned; the split is handled later.
    """
    if isinstance(subsets, str):
        subsets = [subsets]

    result = {}
    for s in subsets:
        if s == "test":
            result[s] = __test_path
        else:   # train or val
            result[s] = __train_path   # same file, split decided in get_dataset

    if len(result) == 1:
        return list(result.values())[0]
    elif all(isinstance(v, Path) for v in result.values()):
        return list(result.values())
    else:
        return result   # dict with subset names


if __name__ == "__main__":
    
    build_dataset()
    labels = create_labels()
    
    mean, std = calculate_statistics()
    print(f"Mean (RGB): {mean}, Std (RGB): {std}")

    train_data, train_labels = get_dataset("train_val")
    print(f"Train data: {train_data.shape}, labels: {train_labels.shape}")

    test_data, test_labels = get_dataset("test")
    print(f"Test data: {test_data.shape}, labels: {test_labels.shape}")
    
    groups = get_groups()
    print('Groups : ')
    for label, val in groups.items():
        print('--', label, ':', len(val))