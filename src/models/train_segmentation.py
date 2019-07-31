import os

import numpy as np
import torch
from torch import nn

from architectures.unet_1 import NestedUNet as Model
from dataset import SegmentationDataset
from losses import bce_dice_loss, dice_coeff


def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    nb_samples = 0
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = bce_dice_loss(output, target, reduction="sum")
        epoch_loss += loss.item()
        loss = loss / len(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
                epoch,
                nb_samples,
                len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                loss.item(),
            ),
            end="\r",
        )

    epoch_loss /= len(train_loader.dataset)
    print(
        "Train Epoch: {} [{}/{} ({:.0f}%)], Average Loss: {:.6f}".format(
            epoch, nb_samples, len(train_loader.dataset), 100.0, epoch_loss
        )
    )
    return epoch_loss


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_dice = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += bce_dice_loss(output, target, reduction="sum").item()
            test_dice += dice_coeff(output, target, hard=True, reduction="sum").item()

    test_loss /= len(test_loader.dataset)
    test_dice /= len(test_loader.dataset)
    print("Test set: Average score: {:.6f} (loss: {:.6f})".format(test_dice, test_loss))
    return test_loss, test_dice


def checkpoint(model, test_dice, optimizer, epoch, input_size, weight_decay, infos=""):
    file_name = "{}_dice={:.3f}_{}_ep={}_{}_wd={}_{}.pth".format(
        model.__class__.__name__,
        test_dice,
        optimizer.__class__.__name__,
        epoch,
        input_size,
        weight_decay,
        infos,
    )
    path = os.path.join("../../models/", file_name)
    if test_dice > 0.47 and not os.path.isfile(path):
        torch.save(model.state_dict(), path)
        print("Saved: ", file_name)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparams
    batch_size = 8
    epochs = 40
    input_size = (216, 320)
    weight_decay = 1e-4
    print(f"Batch size: {batch_size}, input size: {input_size}, wd: {weight_decay}")

    # Create datasets
    train_indices = np.load("../../data/processed/train_indices.npy")
    test_indices = np.load("../../data/processed/test_indices.npy")
    # valid_indices = np.load("../../data/processed/valid_indices.npy")

    # Merge train and test
    # train_indices = np.concatenate((train_indices, test_indices))
    # test_indices = valid_indices

    # Make sure there's no overlap
    assert not set(train_indices) & set(test_indices)

    # Datasets
    train_set = torch.utils.data.Subset(
        SegmentationDataset(
            "../../data/raw/training_set/", input_size=input_size, train_mode=True
        ),
        train_indices,
    )
    test_set = torch.utils.data.Subset(
        SegmentationDataset(
            "../../data/raw/training_set/", input_size=input_size, train_mode=True
        ),
        test_indices,
    )
    print("Training set size: ", len(train_set))
    print("Test set size : ", len(test_set))
    print("Total: ", len(train_set) + len(test_set))

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = Model().to(device)
    print(Model.__name__)

    # he initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-1, momentum=0.9, weight_decay=weight_decay
    )
    print("Optimizer: ", optimizer.__class__.__name__)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 14, 20, 25, 30, 35], gamma=0.1
    )

    train_loss_history = list()
    test_loss_history = list()
    test_dice_history = list()

    for epoch in range(1, epochs + 1):
        print("################## EPOCH {}/{} ##################".format(epoch, epochs))

        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group["lr"])

        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_dice = validate(model, device, test_loader)

        scheduler.step()

        # Save model
        if epoch > 1 and test_dice > max(test_dice_history):
            checkpoint(
                model,
                test_dice,
                optimizer,
                epoch,
                input_size,
                weight_decay,
                infos="bce_dice_loss",
            )

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_dice_history.append(test_dice)

        # # Save history at each epoch (overwrite previous history)
        history = [train_loss_history, test_loss_history, test_dice_history]
        np.save("history.npy", np.array(history))


if __name__ == "__main__":
    main()
