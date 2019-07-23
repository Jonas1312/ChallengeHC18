import os

import numpy as np
import torch

from architectures.unet import UNet as Model
from dataset import SegmentationDataset
from losses import dice_loss


def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    nb_samples = 0
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = dice_loss(output, target)
        epoch_loss += loss.item() * len(data)

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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += dice_loss(output, target, smooth=0.0).item() * len(data)

    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average score: {:.6f} (loss: {:.6f})".format(
            1.0 - test_loss, test_loss
        )
    )
    return test_loss


def checkpoint(model, test_loss, optimizer, epoch, input_size, weight_decay, infos=""):
    file_name = "{}_loss={:.2f}_{}_ep={}_{}_wd={}_{}.pth".format(
        Model.__name__,
        test_loss,
        optimizer.__class__.__name__,
        epoch,
        input_size,
        weight_decay,
        infos,
    )
    path = os.path.join("../../models/", file_name)
    if test_loss < 0.7 and not os.path.isfile(path):
        torch.save(model.state_dict(), path)
        print("Saved: ", file_name)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparams
    batch_size = 16
    epochs = 60
    input_size = (216, 320)
    weight_decay = 0
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
            "../../data/raw/training_set/",
            input_size=input_size,
            random_transforms=True,
        ),
        train_indices,
    )
    test_set = torch.utils.data.Subset(
        SegmentationDataset(
            "../../data/raw/training_set/",
            input_size=input_size,
            random_transforms=False,
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

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    print("Optimizer: ", optimizer.__class__.__name__)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4, 6], gamma=0.1
    )

    train_loss_history = list()
    test_loss_history = list()

    for epoch in range(1, epochs + 1):
        print("################## EPOCH {}/{} ##################".format(epoch, epochs))

        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group["lr"])

        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = validate(model, device, test_loader)

        scheduler.step()

        # Save model
        if epoch > 1 and test_loss < min(test_loss_history):
            checkpoint(
                model, test_loss, optimizer, epoch, input_size, weight_decay, infos=""
            )

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # # Save history at each epoch (overwrite previous history)
        history = [train_loss_history, test_loss_history]
        np.save("history.npy", np.array(history))


if __name__ == "__main__":
    main()
