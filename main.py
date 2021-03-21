import argparse

import pandas as pd

from feature_engineering import feature_engineering, get_dataloaders, get_test_data
from training import train, run_kaggle_submission
import constants as CN

parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_false")
parser.add_argument("-n", default=100, type=int)
parser.add_argument("--kaggle", action="store_true", default=False)
args = parser.parse_args()

train_df = pd.read_csv(CN.TRAIN_FILE)
train_df = feature_engineering(train_df)
train_loader, val_loader = get_dataloaders(train_df=train_df)

test_df = pd.read_csv(CN.TEST_FILE)
test_df = feature_engineering(test_df)
test_df, test_tensors = get_test_data(test_df)

lr = 0.001
n_epochs = args.n
dropout_rate = 0.0
regul = 0.00
submit_to_kaggle = args.kaggle
if submit_to_kaggle:
    print("Kaggle submission is enabled")
else:
    print("Kaggle submission is disabled")
enable_tboard = args.t
for x, _ in train_loader:
    input_dim = x.size(1)
    break

layers_list = [
    (1000, 500, 100, 50),
    (100, 500, 100, 50),
    (100, 50, 20)
]

for layers in layers_list:
    comment = f"Adam_{lr}_{layers}_{CN.BATCH_SIZE}_relu_regul_{regul}_fe"
    print(f"Model running : {comment}")
    model = train(layers=layers,
                  input_dim=input_dim,
                  lr=lr,
                  n_epochs=n_epochs,
                  regul=regul,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  enable_tboard=enable_tboard,
                  comment=comment
                  )

    run_kaggle_submission(model,
                          test_tensors=test_tensors,
                          test_df=test_df,
                          comment=comment,
                          submit=submit_to_kaggle)
