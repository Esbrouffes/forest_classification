from typing import Tuple

import pandas as pd
import torch


def load_training_data(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the filename of the training data, return it
    under the form (samples, labels) as pytorch tensors
    """
    df = pd.read_csv(filename)
    labels = df["Cover_Type"]
    # The Id column is useless
    samples = df.drop(columns=["Cover_Type", "Id"])
    return torch.Tensor(samples.values), torch.Tensor(labels.values)


def load_test_data(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the filename of the test data, return it as pytorch tensors
    """
    df = pd.read_csv(filename)
    # The Id column is useless
    df = df.drop(columns="Id")
    return torch.Tensor(df.values)
