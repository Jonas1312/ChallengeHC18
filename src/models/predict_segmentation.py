import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from architectures.custom_unet import U_Net as Model
from dataset import SegmentationDataset
from losses import dice_coeff
from ellipse_fitting import opencv_fitEllipse


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    weights_name = "U_Net_dice=0.476_SGD_ep=30_(216, 320)_wd=0.0001_bce_dice_loss.pth"

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
            if isinstance(output, list):
                output = output[0]
            dice = dice_coeff(output, target).item()
            print("dice: ", dice)

            output = torch.sigmoid(output)
            mask = output.cpu().numpy()[0, 0, :, :]
            (xx, yy), (MA, ma), angle = opencv_fitEllipse(mask)

            output = output[0, 0].cpu().numpy()

            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

            cv2.ellipse(
                output,
                (int(yy), int(xx)),
                (int(ma / 2), int(MA / 2)),
                -angle,
                0,
                360,
                color=(1, 0, 0),
                thickness=2,
            )

            plt.imshow(data[0, 0].cpu(), cmap="gray")
            plt.figure()
            plt.imshow(target[0, 0].cpu(), cmap="gray")
            plt.figure()
            plt.imshow(mask, cmap="gray")
            plt.figure()
            plt.imshow(output)
            plt.show()

            ret = input("Continue? ")
            if "y" not in ret:
                break


if __name__ == "__main__":
    main()
