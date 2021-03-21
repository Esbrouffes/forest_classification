import argparse

from training import train, run_kaggle_submission
from load_data import train_loader, val_loader, test_df, test_tensors
import constants as CN

parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_false")
parser.add_argument("-n", default=100, type=int)
parser.add_argument("--kaggle", action="store_true", default=False)
args = parser.parse_args()

lr = 0.001
n_epochs = args.n
dropout_rate = 0.0
regul = 0.00
submit_to_kaggle = args.kaggle
enable_tboard = args.t
layers = [
    # (1000, 500, 100, 50),
    # (100, 500, 100, 50),
    # (50, 100, 100, 50),
    (500, 100, 10),
    (50, 100, 10, 10),
    # (400, 100, 50, 10),
    # (1000, 500, 100, 50),
    # (100, 100, 50, 20)
]
for layer in layers:
    comment = f"Adam_{lr}_{layers}_{CN.BATCH_SIZE}_relu_regul_{regul}"
    print(f"Model running : {comment}")
    model = train(layer,
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
                          submit=True)
