def add_ellipses_csv():
    """Add ellipses parameters to training_set_pixel_size_and_HC.csv"""
    import numpy as np
    import pandas as pd
    import os
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    df = pd.read_csv("../../data/raw/training_set_pixel_size_and_HC.csv")

    centers_x = list()
    centers_y = list()
    axes_a = list()
    axes_b = list()
    angles = list()

    for i, row in df.iterrows():
        filename = row["filename"]
        print("i: ", i, end="\r")
        img = Image.open(
            os.path.join(
                "../../data/raw/training_set/",
                filename.replace(".png", "_Annotation.png"),
            )
        )
        img = np.array(img)
        points = np.argwhere(img > 127)
        (xx, yy), (MA, ma), angle = cv2.fitEllipseDirect(points)

        factor = row["pixel size(mm)"]

        center_x_mm = factor * yy
        center_y_mm = factor * xx
        semi_axes_a_mm = factor * ma / 2
        semi_axes_b_mm = factor * MA / 2
        angle_rad = (-angle * np.pi / 180) % np.pi
        # print(center_x_mm, center_y_mm, semi_axes_a_mm, semi_axes_b_mm, angle_rad)

        centers_x.append(center_x_mm)
        centers_y.append(center_y_mm)
        axes_a.append(semi_axes_a_mm)
        axes_b.append(semi_axes_b_mm)
        angles.append(angle_rad)

        h = (semi_axes_a_mm - semi_axes_b_mm) ** 2 / (
            semi_axes_a_mm + semi_axes_b_mm
        ) ** 2
        circ = (
            np.pi
            * (semi_axes_a_mm + semi_axes_b_mm)
            * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        )

        assert np.abs(circ - row["head circumference (mm)"]) < 0.1

        # print("circ: ", circ)
        # print("true circ: ", row["head circumference (mm)"])

        # plt.imshow(img)
        # plt.show()

    df["center_x_mm"] = centers_x
    df["center_y_mm"] = centers_y
    df["semi_axes_a_mm"] = axes_a
    df["semi_axes_b_mm"] = axes_b
    df["angle_rad"] = angles
    print(df)
    df.to_csv(
        "../../data/processed/training_set_pixel_size_and_HC_and_ellipses.csv",
        index=False,
    )


def generate_test_valid_indices():
    """To be used with torch.utils.data.Subset"""
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split

    output_dir = "../../data/processed/"

    X = np.arange(999)

    X_left, X_test, y_left, _ = train_test_split(X, np.ones_like(X), test_size=0.15)
    X_train, X_valid, _, _ = train_test_split(
        X_left, y_left, test_size=0.05 / (1 - 0.15)
    )

    train_indices = X_train
    test_indices = X_test
    valid_indices = X_valid

    assert len(train_indices) + len(test_indices) + len(valid_indices) == 999

    assert not set(train_indices) & set(test_indices)
    assert not set(test_indices) & set(valid_indices)
    assert not set(train_indices) & set(valid_indices)

    np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    np.save(os.path.join(output_dir, "valid_indices.npy"), valid_indices)


if __name__ == "__main__":
    to_run = (add_ellipses_csv,)
    for func in to_run:
        ret = input(f'Run "{func.__name__}"? (y/n) ')
        if "y" in ret:
            func()
            break
