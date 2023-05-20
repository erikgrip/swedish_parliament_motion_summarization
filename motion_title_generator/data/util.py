import torch
from torch.utils.data import random_split


def split_data(df, train_frac, val_frac, seed):
    """Return train, validation and test dataframes"""
    train_size = round(len(df) * train_frac)
    val_size = round(len(df) * val_frac)
    test_size = len(df) - train_size - val_size

    if (train_size + val_size + test_size) != len(df):
        raise ValueError(f"Invalid train/val fractions: {train_frac}, {val_frac}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    return random_split(
        dataset=df,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
