from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


@dataclass
class DatasetInfo:
    input_shape: Tuple[int, ...]
    num_classes: int


def _build_adult_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    test_size = cfg["dataset"]["adult"]["test_size"]
    val_size = cfg["dataset"]["adult"]["val_size"]
    random_state = cfg["dataset"]["adult"]["random_state"]

    csv_path = cfg["dataset"]["adult"]["csv_path"]

    # Try to read local CSV; if missing, fall back to OpenML download.
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        from sklearn.datasets import fetch_openml

        print(
            f"Adult CSV not found at '{csv_path}'. "
            "Downloading 'adult' dataset from OpenML instead..."
        )
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame
        # Standard OpenML adult dataset uses 'class' as target with values '<=50K' and '>50K'.
        df = df.rename(columns={"class": "income"})

    # Assume classic Adult dataset column names; adjust if needed.
    target_col = "income"
    y = (df[target_col].str.contains(">50K")).astype(int).values
    X = df.drop(columns=[target_col])

    # Treat both 'object' and 'category' dtypes as categorical.
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    X_cat = ohe.fit_transform(X[categorical_cols]) if categorical_cols else np.empty(
        (len(X), 0)
    )
    X_num = scaler.fit_transform(X[numeric_cols]) if numeric_cols else np.empty(
        (len(X), 0)
    )

    X_all = np.concatenate([X_num, X_cat], axis=1).astype("float32")

    # First split off test, then split remaining into train/val.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval,
    )

    def to_loader(
        X_arr: np.ndarray, y_arr: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X_arr), torch.from_numpy(y_arr.astype("int64"))
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg["training"]["num_workers"],
        )

    batch_size = cfg["training"]["batch_size"]
    train_loader = to_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = to_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = to_loader(X_test, y_test, batch_size, shuffle=False)

    info = DatasetInfo(input_shape=(X_all.shape[1],), num_classes=2)
    return train_loader, val_loader, test_loader, info


def _build_cifar10_loaders(
    cfg,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    root = cfg["dataset"]["cifar10"]["root"]
    val_size = cfg["dataset"]["cifar10"]["val_size"]
    random_state = cfg["dataset"]["cifar10"]["random_state"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    full_train = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    num_train = len(full_train)
    indices = np.arange(num_train)
    val_count = int(num_train * val_size)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_subset = torch.utils.data.Subset(full_train, train_idx.tolist())
    val_subset = torch.utils.data.Subset(full_train, val_idx.tolist())

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    info = DatasetInfo(input_shape=(3, 32, 32), num_classes=10)
    return train_loader, val_loader, test_loader, info


def _build_pcam_loaders(
    cfg,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    from torchvision.datasets import PCAM

    root = cfg["dataset"]["pcam"]["root"]
    split_train = cfg["dataset"]["pcam"]["split_train"]
    split_val = cfg["dataset"]["pcam"]["split_val"]
    split_test = cfg["dataset"]["pcam"]["split_test"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    train_set = PCAM(root=root, split=split_train, download=True, transform=transform)
    val_set = PCAM(root=root, split=split_val, download=True, transform=transform)
    test_set = PCAM(root=root, split=split_test, download=True, transform=transform)

    batch_size = cfg["training"]["batch_size"]
    # On some platforms (e.g. Windows with newer Python), multi-processing
    # DataLoader workers can be problematic. Allow overriding workers for PCam.
    num_workers = cfg["training"].get(
        "num_workers_pcam", cfg["training"].get("num_workers", 0)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # PCam images are 96x96 RGB
    info = DatasetInfo(input_shape=(3, 96, 96), num_classes=2)
    return train_loader, val_loader, test_loader, info


def build_dataloaders(cfg, dataset_name: str):
    """Entry point to build dataloaders and dataset metadata."""
    if dataset_name == "adult":
        return _build_adult_loaders(cfg)
    if dataset_name == "cifar10":
        return _build_cifar10_loaders(cfg)
    if dataset_name == "pcam":
        return _build_pcam_loaders(cfg)
    raise ValueError(f"Unknown dataset name: {dataset_name}")


