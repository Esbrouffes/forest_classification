import argparse

import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter

from load_data import train_loader, val_loader, test_loader
from mlp import MLP
import constants as CN

parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_false")
args = parser.parse_args()

lr = 0.001
n_epochs = 150
drop_rate = 0.0
layers = (100, 50, 20)
comment = f"Adam_{lr}_{layers}_{CN.BATCH_SIZE}_relu_{drop_rate}_drop"

if not args.t:
    writer = SummaryWriter("runs/trash")
    print(f"Running without logging")
else:
    writer = SummaryWriter(comment=comment)
    print(f"Running {comment}")
model = MLP(CN.INPUT_DIM, CN.N_CLASS, layers=layers, drop_rate=drop_rate)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

for epoch in tqdm.tqdm(range(n_epochs)):
    epoch_loss = 0
    total_sample = 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_function(out, y.long())
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
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
    writer.add_scalar("Accuracy", total_correct / total_sample, epoch)
    writer.add_scalar("Loss/val", epoch_loss, epoch)
writer.close()
