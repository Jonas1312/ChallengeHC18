import torch
from torch.nn import functional as F

EPS = 1e-7


def dice_coeff(pred, target, smooth=0.0, hard=False, reduction="mean"):
    pred = torch.sigmoid(pred)

    img_flat = pred.view(pred.size(0), -1)
    mask_flat = target.view(target.size(0), -1)

    if hard:
        img_flat = torch.round(img_flat)

    intersection = (img_flat * mask_flat).sum(dim=-1)

    dice = (2.0 * intersection + smooth) / (
        img_flat.sum(dim=-1) + mask_flat.sum(dim=-1) + smooth + EPS
    )
    if reduction == "mean":
        return dice.mean()
    if reduction == "sum":
        return dice.sum()
    return dice


def dice_loss(pred, target, smooth=1.0, reduction="mean"):
    dice_per_image = dice_coeff(pred, target, smooth=smooth, reduction="none")
    loss_per_image = 1.0 - dice_per_image
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def log_dice_loss(pred, target, smooth=1.0, reduction="mean"):
    loss_per_image = -torch.log(
        dice_coeff(pred, target, smooth=smooth, reduction="none")
    )
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def bce_loss(pred, target, reduction="mean"):
    """Balanced binary cross entropy"""
    beta = (1 - target).sum() / target.numel()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    loss = (beta.sqrt() * target + (1 - beta).sqrt() * (1 - target)) * loss

    loss_per_image = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def focal_loss(pred, target, gamma=2, reduction="mean"):
    y_hat = torch.sigmoid(pred)

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    loss = ((1 - y_hat) ** gamma * target + (y_hat) ** gamma * (1 - target)) * loss

    loss_per_image = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def bce_dice_loss(pred, target, reduction="mean"):
    return bce_loss(pred, target, reduction=reduction) + dice_loss(
        pred, target, reduction=reduction
    )


def exp_log_loss(pred, target, wdice=0.8, wcross=0.2, gamma=0.3, reduction="mean"):
    """https://arxiv.org/pdf/1809.00076.pdf"""
    l_dice = log_dice_loss(pred, target, reduction="none") ** gamma
    l_cross = bce_loss(pred, target, reduction="none") ** gamma

    loss_per_image = wdice * l_dice + wcross * l_cross
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def tversky_coeff(
    pred, target, alpha=0.3, beta=0.7, smooth=0.0, hard=False, reduction="mean"
):
    """https://arxiv.org/pdf/1706.05721.pdf
    α and β control the magnitude of penalties for FPs and FNs, respectively
    α = β = 0.5 => dice coeff
    α = β = 1   => tanimoto coeff
    α + β = 1   => F beta coeff
    """
    pred = torch.sigmoid(pred)

    img_flat = pred.view(pred.size(0), -1)
    mask_flat = target.view(target.size(0), -1)

    if hard:
        img_flat = torch.round(img_flat)

    intersection = (img_flat * mask_flat).sum(dim=-1)
    fps = (img_flat * (1 - mask_flat)).sum(dim=-1)
    fns = ((1 - img_flat) * mask_flat).sum(dim=-1)

    denominator = intersection + alpha * fps + beta * fns
    tversky_per_image = (intersection + smooth) / (denominator + smooth)
    if reduction == "mean":
        return tversky_per_image.mean()
    if reduction == "sum":
        return tversky_per_image.sum()
    return tversky_per_image


def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0, reduction="mean"):
    """https://arxiv.org/pdf/1706.05721.pdf
    α and β control the magnitude of penalties for FPs and FNs, respectively
    α = β = 0.5 => dice coeff
    α = β = 1   => tanimoto coeff
    α + β = 1   => F beta coeff
    """
    tversky_per_image = tversky_coeff(
        pred,
        target,
        alpha=alpha,
        beta=beta,
        smooth=smooth,
        hard=False,
        reduction="none",
    )

    loss_per_image = 1.0 - tversky_per_image
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def focal_tversky_loss(pred, target, gamma=4.0 / 3, reduction="mean"):
    """https://arxiv.org/pdf/1810.07842.pdf"""
    loss_per_image = tversky_loss(pred, target, reduction="none") ** (1.0 / gamma)
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


if __name__ == "__main__":

    def main():
        pred = torch.zeros((3, 1, 5, 5)) - 100
        target = torch.zeros_like(pred)

        pred[0, 0, 0, 0] = 100
        target[0, 0, 0, 0] = 1
        pred[1, 0, 0, 0] = 100  # fp
        target[1, 0, 0, 0] = 0
        pred[2, 0, 0, 0] = -100  # fn
        target[2, 0, 0, 0] = 1

        print(tversky_loss(pred, target, reduction="dd"))
        print(dice_loss(pred, target, reduction="dd"))

    main()
