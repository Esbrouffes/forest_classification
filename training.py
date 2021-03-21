import os

import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter

from mlp import MLP
import constants as CN


def train(layers, input_dim, lr, n_epochs, regul, train_loader, val_loader, enable_tboard, comment):
    if not enable_tboard:
        writer = SummaryWriter("runs/trash")
        print(f"Running without tensorboard logging")
    else:
        print(f"Running with tensorboard logging")
        writer = SummaryWriter(comment=comment)
    model = MLP(input_dim=input_dim,
                out_dim=CN.N_CLASS,
                layers=layers)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regul)
    loss_function = nn.CrossEntropyLoss()

    print(f"Training on {n_epochs} epochs")
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
    writer.close()
    return model


def run_kaggle_submission(model, test_tensors, test_df, comment, submit=False):
    print("Validation on Kaggle data")
    with torch.set_grad_enabled(False):
        model.eval()
        output = model(test_tensors)
        output_classes = output.argmax(dim=1)
    # The position of the max starts at index 0
    output_classes += 1
    test_df["Cover_Type"] = output_classes
    test_df.to_csv("submission.csv", header=True, columns=["Id", "Cover_Type"], index=False)
    if submit:
        os.system(CN.KAGGLE_COMMAND.format(comment))
        print("\n")
