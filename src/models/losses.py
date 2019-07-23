def dice_loss(pred, target, smooth=1.0):
    img_flat = pred.view(-1)
    mask_flat = target.view(-1)

    intersection = (img_flat * mask_flat).sum()

    dice = (2.0 * intersection + smooth) / (img_flat.sum() + mask_flat.sum() + smooth)

    return 1 - dice
