import torch
from torch.utils.data import random_split

from utils.log import logger


def split_data(df, train_frac, val_frac, seed):
    """Return train, validation and test dataframes"""
    train_size = round(len(df) * train_frac)
    val_size = round(len(df) * val_frac)
    test_size = len(df) - train_size - val_size

    if (train_size + val_size + test_size) != len(df):
        raise ValueError(f"Invalid train/val fractions: {train_frac}, {val_frac}")
    logger.info(
        "Data split sizes -- train: %s, val: %s, test: %s",
        train_size,
        val_size,
        test_size,
    )

    return random_split(
        dataset=df,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
