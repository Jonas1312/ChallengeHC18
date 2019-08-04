import os

import numpy as np
import pandas as pd
import torch

from architectures.custom_unet import U_Net as Model
from dataset import SegmentationDataset
from ellipse_fitting import opencv_fitEllipse


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    weights_name = (
        "U_Net_dice=0.483_SGD_ep=24_(216, 320)_wd=0.0001_final_bce_dice_loss.pth"
    )

    input_size = (216, 320)

    test_set = SegmentationDataset(
        "../../data/raw/test_set/", input_size=input_size, train_mode=False
    )
    print("Test set size : ", len(test_set))

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=32, shuffle=False, pin_memory=True
    )

    model = Model().to(device)
    print(Model.__name__)
    model.load_state_dict(torch.load(os.path.join("../../models/", weights_name)))
    model.eval()

    centers_x = list()
    centers_y = list()
    axes_a = list()
    axes_b = list()
    angles = list()

    nb_samples = 0
    with torch.no_grad():
        for data, factors in test_loader:
            print(nb_samples)
            nb_samples += len(data)

            data = data.to(device)
            output = model(data)
            if isinstance(output, list):
                output = output[0]

            output = torch.sigmoid(output)

            for batch_index in range(output.size(0)):
                mask = output.cpu().numpy()[batch_index, 0, :, :]
                (xx, yy), (MA, ma), angle = opencv_fitEllipse(mask)

                assert 540 / mask.shape[0] == 800 / mask.shape[1]

                factor = factors[batch_index].item() * 540 / mask.shape[0]

                center_x_mm = factor * yy
                center_y_mm = factor * xx
                semi_axes_a_mm = factor * ma / 2
                semi_axes_b_mm = factor * MA / 2
                angle_rad = (-angle * np.pi / 180) % np.pi

                centers_x.append(center_x_mm)
                centers_y.append(center_y_mm)
                axes_a.append(semi_axes_a_mm)
                axes_b.append(semi_axes_b_mm)
                angles.append(angle_rad)

    df = pd.read_csv("../../data/raw/test_set_pixel_size.csv")
    df = df.drop(columns="pixel size(mm)")
    df["center_x_mm"] = centers_x
    df["center_y_mm"] = centers_y
    df["semi_axes_a_mm"] = axes_a
    df["semi_axes_b_mm"] = axes_b
    df["angle_rad"] = angles
    print(df)
    csv_name = weights_name.replace(".pth", ".csv")
    df.to_csv(os.path.join("../../submission_csv/", csv_name), index=False)


if __name__ == "__main__":
    main()
