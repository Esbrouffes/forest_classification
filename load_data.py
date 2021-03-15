from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import constants as CN


def load_training_data(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the filename of the training data, return it
    under the form (samples, labels) as pytorch tensors
    """
    df = pd.read_csv(filename)
    labels = df["Cover_Type"]
    labels_array = labels.values
    labels_array -= 1
    # The Id column is useless
    samples = df.drop(columns=["Cover_Type", "Id"])
    return torch.Tensor(samples.values), torch.Tensor(labels_array)


def load_test_data(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the filename of the test data, return it as pytorch tensors
    """
    df = pd.read_csv(filename)
    # The Id column is useless
    df = df.drop(columns="Id")
    return torch.Tensor(df.values)


train_samples, train_labels = load_training_data(CN.TRAIN_FILE)
X_train, X_val, y_train, y_val = train_test_split(train_samples, train_labels, test_size=0.20)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=CN.BATCH_SIZE, drop_last=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=CN.BATCH_SIZE, drop_last=True)

test_samples = load_test_data(CN.TEST_FILE)
test_dataset = TensorDataset(test_samples)
test_loader = DataLoader(test_dataset, batch_size=CN.BATCH_SIZE, drop_last=True)
