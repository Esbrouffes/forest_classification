import torch
import pandas as pd

import constants as CN


def read_csv(filename: str) -> torch.Tensor:
    df = pd.read_csv(filename)
    return torch.Tensor(df.values)


train_data = read_csv(CN.TRAIN_FILE)
test_data = read_csv(CN.TEST_FILE)
print(train_data.shape)
print(test_data.shape)
