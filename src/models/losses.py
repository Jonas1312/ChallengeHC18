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


# def bce_loss(pred, target, weight):
#     bce_pos = target * pred
#     bce_neg = (1 - target) * (1 - pred)
#     loss = -torch.log(bce_pos + bce_neg)
#     loss = loss * target * weight + loss * (1 - target) * (1 - weight)
#     loss = torch.mean(loss)
#     return loss


def focal_loss(pred, target, gamma=2, reduction="mean"):
    # pred = torch.sigmoid(pred)
    # pred_pos = target * pred
    # pred_neg = (1 - target) * (1 - pred)
    # pt = pred_pos + pred_neg
    # loss = -(1 - pt) ** gamma * torch.log(pt)
    y_hat = torch.sigmoid(pred)
    neg_pred = -torch.abs(pred)
    # log_exp = (1 + (-pred).exp()).log()
    log_exp = pred.clamp(min=0) - pred * target + (1 + neg_pred.exp()).log()
    loss_pos = (1 - y_hat) ** gamma * target
    loss_neg = -y_hat ** gamma * (1 - target) * pred
    loss = (loss_pos + loss_neg) * log_exp

    loss_per_sample = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    return loss_per_sample


def lovasz_hinge(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


def lovasz_hinge_flat(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * torch.Tensor(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), torch.Tensor(grad))
    return loss


def lovasz_grad(gt_sorted):
    r"""
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + EPS)
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


if __name__ == "__main__":

    def main():
        img = torch.zeros((3, 1, 5, 5))
        target = torch.zeros((3, 1, 5, 5))

        target[0, 0, 0, 0] = 1
        img[0, 0, 0, 0] = 1
        target[1, 0, 0, 1] = 1
        img[1, 0, 0, 2] = 1
        img[1, 0, 0, 1] = 1
        target[2, 0, 0, 1] = 1

    main()
