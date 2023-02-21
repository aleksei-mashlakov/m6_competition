import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import Tensor


class RPSLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(RPSLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor):
        # y_pred = F.softmax(input, dim=-1)
        #         outcome = F.one_hot(target.to(torch.int64), num_classes=5)
        diff = torch.cumsum(input, dim=-1) - torch.cumsum(target, dim=-1)
        loss = torch.mean(((diff) ** 2))  # , dim=0)
        return loss


def batchify_data(x_data, y_data, batch_size: int = 25):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append(
            {
                "x": x_data[i : i + batch_size, :, :, :],
                "y": y_data[i : i + batch_size, :, :, :],
            }
        )
    return batches


def train_model(train_data, dev_data, model, optimizer, n_epochs=30):
    """Train a model for N epochs given data and hyper-params."""

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        losses = run_epoch(train_data, model.train(), optimizer)
        #         [print('Train | loss{}: {:.6f} '.format(i, losses[i])) for i in range(len(losses))]
        print("Train | loss: {:.6f} ".format(np.mean(losses)))

        # Run **validation**
        val_losses = run_epoch(dev_data, model.eval(), optimizer)
        print("Valid | loss: {:.6f} ".format(np.mean(val_losses)))

    #         [print('Valid | loss{}: {:.6f} '.format(i, losses[i])) for i in range(len(val_losses))]
    return model


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, accuracy"""
    # Gather losses
    losses = [[] for i in range(model.in_channels)]

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm.tqdm(data):

        # Grab x and y
        x, y = batch["x"], batch["y"]

        # Get output predictions
        preds = model(x)

        # Compute losses
        for i in range(model.in_channels):
            criterion = RPSLoss()
            # criterion = torch.nn.CrossEntropyLoss()
            target = F.one_hot(
                y[:, i, 0, 0].to(torch.int64), num_classes=5
            )  # .to(torch.float)
            loss = criterion(preds[i], target)
            losses[i].append(loss)

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            joint_loss = torch.mean(
                torch.stack([loss[-1] for loss in losses])
            )  # , axis=1)
            #             joint_loss = torch.stack([loss[-1] for loss in losses])
            joint_loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    #     print(np.mean(torch.stack(losses[0]).detach().numpy()))
    avg_loss = [np.mean(torch.stack(loss).detach().numpy()) for loss in losses]
    return avg_loss
