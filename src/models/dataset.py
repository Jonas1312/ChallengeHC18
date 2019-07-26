import os
import os.path
import random

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

from torchvision import transforms
from torchvision.transforms import functional as F


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    return pil_loader(path)


class SegmentationDataset(data.Dataset):
    def __init__(self, root, input_size, train_mode):
        super().__init__()
        self.root = root
        self.input_size = input_size
        self.train_mode = train_mode

        self.samples = self.make_dataset(self.root)
        self.loader = default_loader

        if train_mode:
            self.df = pd.read_csv(
                "../../data/processed/training_set_pixel_size_and_HC_and_ellipses.csv"
            )
        else:
            self.df = pd.read_csv("../../data/raw/test_set_pixel_size.csv")

    def __getitem__(self, index):
        img_name = self.samples[index]
        img = self.loader(os.path.join(self.root, img_name))

        if self.train_mode:
            mask = self.loader(
                os.path.join(self.root, img_name.replace(".png", "_Annotation.png"))
            )
            img, mask = self.apply_transform(img, mask)
            return img, mask

        img = self.apply_transform(img)
        factor = self.df.iloc[index]["pixel size(mm)"]
        return img, factor

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def make_dataset(directory):
        images = os.listdir(directory)

        images = sorted([x for x in images if x.lower().endswith("hc.png")])

        # assert len(images) == 999

        return images

    def apply_transform(self, image, mask=None):

        # Grayscale
        image = transforms.functional.to_grayscale(image)
        if self.train_mode:
            mask = transforms.functional.to_grayscale(mask)

        # Resize
        resize = transforms.Resize(size=self.input_size)
        image = resize(image)
        if self.train_mode:
            mask = resize(mask)

        if self.train_mode:
            # Random affine
            random_aff = transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                resample=3,
                fillcolor=0,
            )
            ret = random_aff.get_params(
                random_aff.degrees,
                random_aff.translate,
                random_aff.scale,
                random_aff.shear,
                image.size,
            )
            image = F.affine(
                image,
                *ret,
                resample=random_aff.resample,
                fillcolor=random_aff.fillcolor
            )
            mask = F.affine(
                mask, *ret, resample=random_aff.resample, fillcolor=random_aff.fillcolor
            )

            # Random horizontal flipping
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

        # Transform to tensor
        image = F.to_tensor(image)
        if self.train_mode:
            mask = F.to_tensor(mask)

        # Binarize mask
        if self.train_mode:
            mask = torch.where(
                mask > 0.1, torch.tensor(1.0), torch.tensor(0.0)
            )  # TODO: tune threshold

            return image, mask
        return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def main():
        dataset = SegmentationDataset(
            "../../data/raw/training_set/", input_size=(216, 320), train_mode=True
        )
        print(len(dataset))
        for i in range(len(dataset)):
            img_, mask_ = dataset[i]
            print(img_.size())
            print(mask_.size())
            plt.imshow(img_[0, :, :])
            plt.figure()
            plt.imshow(mask_[0, :, :])
            plt.show()

    main()
