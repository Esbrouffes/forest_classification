import argparse
import os
import time

import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter

from load_data import train_loader, val_loader, test_df, test_tensors
from mlp import MLP
import constants as CN

parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_false")
args = parser.parse_args()

lr = 0.001
n_epochs = 200
dropout_rate = 0.0
regul = 0.00


def train(layers, lr, n_epochs, regul):
    comment = f"Adam_{lr}_{layers}_{CN.BATCH_SIZE}_relu_regul_{regul}"
    if not args.t:
        writer = SummaryWriter("runs/trash")
        print(f"Running without logging")
    else:
        writer = SummaryWriter(comment=comment)
        print(f"Running {comment}")
    model = MLP(CN.INPUT_DIM, CN.N_CLASS, layers=layers)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regul)
    loss_function = nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(n_epochs)):
        epoch_loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_function(out, y.long())
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        epoch_loss /= len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_loss = 0
        total_correct = 0
        total_sample = 0
        with torch.set_grad_enabled(False):
            model.eval()
            for i, (x, y) in enumerate(val_loader):
                optimizer.zero_grad()
                out = model(x)
                output_classes = out.argmax(dim=1)
                correct = (output_classes == y).sum().float()
                total_correct += correct
                total_sample += x.size(0)
                loss = loss_function(out, y.long())
                epoch_loss += loss.item()
        epoch_loss /= len(val_loader)
        writer.add_scalar("Accuracy", total_correct / total_sample, epoch)
        writer.add_scalar("Loss/val", epoch_loss, epoch)

    with torch.set_grad_enabled(False):
        model.eval()
        output = model(test_tensors)
        output_classes = output.argmax(dim=1)
    # The position of the max starts at index 0
    output_classes += 1
    test_df["Cover_Type"] = output_classes
    test_df.to_csv("submission.csv", header=True, columns=["Id", "Cover_Type"], index=False)

    os.system(CN.KAGGLE_COMMAND.format(comment))
    # time.sleep(4)
    # os.system(CN.KAGGLE_SUBMISSIONS)

    writer.close()


layers = [
    # (1000, 500, 100, 50),
    # (100, 500, 100, 50),
    # (50, 100, 100, 50),
    (500, 100, 10),
    # (400, 100, 50, 10),
    # (1000, 500, 100, 50),
    # (100, 100, 50, 20)
]
for layer in layers:
    train(layer)
