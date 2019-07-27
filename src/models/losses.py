import torch
from torch.nn import functional as F

EPS = 1e-7


def dice_loss(pred, target, smooth=1.0, reduction="mean"):
    dice = dice_coeff(pred, target, smooth=smooth, reduction=reduction)
    if reduction == "sum":
        return pred.size(0) - dice
    return 1 - dice


def dice_coeff(pred, target, smooth=0.0, reduction="mean"):
    pred = torch.sigmoid(pred)

    img_flat = pred.view(pred.size(0), -1)
    mask_flat = target.view(target.size(0), -1)

    intersection = (img_flat * mask_flat).sum(dim=-1)

    dice = (2.0 * intersection + smooth) / (
        img_flat.sum(dim=-1) + mask_flat.sum(dim=-1) + smooth + EPS
    )
    if reduction == "mean":
        return dice.mean()
    if reduction == "sum":
        return dice.sum()
    return dice


def bce_loss(pred, target, reduction="mean"):
    """Balanced binary cross entropy"""
    beta = (1 - target).sum() / target.numel()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    loss = (beta * target + (1 - beta) * (1 - target)) * loss

    loss_per_sample = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    return loss_per_sample


def bce_dice_loss(pred, target, reduction="mean"):
    return bce_loss(pred, target, reduction=reduction) + dice_loss(
        pred, target, reduction=reduction
    )


def focal_loss(pred, target, gamma=2, reduction="mean"):
    y_hat = torch.sigmoid(pred)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    loss = ((1 - y_hat) ** gamma * target + (y_hat) ** gamma * (1 - target)) * loss

    loss_per_sample = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    return loss_per_sample


if __name__ == "__main__":

    def main():
        pred = torch.zeros((3, 1, 5, 5))
        target = torch.zeros_like(pred)

        pred[0, 0, 0, 0] = 10
        target[0, 0, 0, 0] = 1
        pred[1, 0, 0, 0] = 1
        target[1, 0, 0, 1] = 1
        pred[2, 0, 0, 0] = -10
        target[2, 0, 0, 0] = 0

        print(bce_loss(pred, target, reduction="dd"))
        print(bce_loss(pred, target, reduction="sum"))
        print(bce_loss(pred, target, reduction="mean"))

    main()
