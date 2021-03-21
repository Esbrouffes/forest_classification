import argparse

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
    epoch_loss /= len(val_loader)
    writer.add_scalar("Accuracy", total_correct / total_sample, epoch)
    writer.add_scalar("Loss/val", epoch_loss, epoch)
writer.close()
