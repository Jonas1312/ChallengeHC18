import numpy as np

import cv2


def opencv_fitEllipse(binary_mask, method="Direct"):

    points = np.argwhere(binary_mask > 0.5)

    if method == "AMS":
        (xx, yy), (MA, ma), angle = cv2.fitEllipseAMS(points)
    elif method == "Direct":
        (xx, yy), (MA, ma), angle = cv2.fitEllipseDirect(points)
    elif method == "Simple":
        (xx, yy), (MA, ma), angle = cv2.fitEllipse(points)
    else:
        raise ValueError("Wrong method")

    return (xx, yy), (MA, ma), angle


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def main():
        img = np.load("./output_samples/3.npy")
        (xx, yy), (MA, ma), angle = opencv_fitEllipse(img)

        draw = np.zeros((*img.shape, 3), dtype=np.uint8)

        draw[img > 0.5] = (255,) * 3

        cv2.ellipse(
            draw,
            (int(yy), int(xx)),
            (int(ma / 2), int(MA / 2)),
            -angle,
            0,
            360,
            color=(255, 0, 0),
            thickness=2,
        )

        plt.imshow(draw)
        plt.show()

    main()
