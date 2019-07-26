import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from architectures.unet1 import UNet1 as Model
from dataset import SegmentationDataset
from losses import dice_coeff


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    weights_name = "UNet_loss=0.63_SGD_ep=5_(216, 320).pth"

    # Hyperparams
    input_size = (216, 320)

    # Create datasets
    test_indices = np.load("../../data/processed/test_indices.npy")

    # Datasets
    test_set = torch.utils.data.Subset(
        SegmentationDataset(
            "../../data/raw/training_set/", input_size=input_size, train_mode=True
        ),
        test_indices,
    )
    print("Test set size : ", len(test_set))

    # Dataloaders
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=True, pin_memory=True
    )

    model = Model().to(device)
    print(Model.__name__)
    model.load_state_dict(torch.load(os.path.join("../../models/", weights_name)))
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            dice = dice_coeff(output, target).item()
            print("dice: ", dice)

            plt.imshow(data[0, 0].cpu(), cmap="gray")
            plt.figure()
            plt.imshow(target[0, 0].cpu(), cmap="gray")
            plt.figure()
            plt.imshow(output[0, 0].cpu(), cmap="gray")
            plt.show()

            ret = input("Continue? ")
            if "y" not in ret:
                break


if __name__ == "__main__":
    main()
