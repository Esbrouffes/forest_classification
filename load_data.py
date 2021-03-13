from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import constants as CN


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


train_samples, train_labels = load_training_data(CN.TRAIN_FILE)
train_dataset = TensorDataset(train_samples, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16)

test_samples = load_test_data(CN.TEST_FILE)
test_dataset = TensorDataset(test_samples)
test_loader = DataLoader(test_dataset, batch_size=16)
