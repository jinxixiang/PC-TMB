import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os


class KLDLoss(nn.Module):
    """KL-divergence loss between attention weight and uniform distribution"""

    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, attn_val, cluster):
        """
        Example:
          Input - attention value = torch.tensor([0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05,
                                0.1, 0.05, 0.1, 0.05, 0.1, 0.05])
                  cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
          Output - 0.0043
        """
        kld_loss = 0
        device = attn_val.device
        # cluster = np.array(cluster)

        for cls in torch.unique(cluster):
            index = torch.where(cluster == cls)[0]
            # HARD CODE - if number of images less than 4 in a cluster then skip
            if len(index) <= 4:
                continue

            att_dist = F.log_softmax(attn_val[0, index], dim=-1)[None]
            cluster_dist = torch.ones(1, len(index)).to(device) / len(index)
            kld_loss += F.kl_div(att_dist, cluster_dist, reduction='batchmean')

        return kld_loss


def smoothcrossentropyloss(pred, gold, n_class=6, smoothing=0.1):
    gold = gold.contiguous().view(-1)
    one_hot = torch.zeros_like(pred).to(gold.device).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred)
    loss = loss.sum(1)
    return loss


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5, reduction="mean"):
    """
    https://github.com/OceanPang/Libra_R-CNN/blob/5d6096f39b90eeafaf3457f5a39572fe5e991808/mmdet/models/losses/balanced_l1_loss.py
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    assert reduction in {"none", "sum", "mean"}

    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta,
    )

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss
    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    https://github.com/OceanPang/Libra_R-CNN/blob/5d6096f39b90eeafaf3457f5a39572fe5e991808/mmdet/models/losses/balanced_l1_loss.py
    """

    def __init__(
            self, alpha=0.5, gamma=1.5, beta=1.0, reduction="mean", loss_weight=1.0
    ):
        super(BalancedL1Loss, self).__init__()
        assert reduction in {"none", "sum", "mean"}
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss_bbox


class FocalLoss(nn.Module):
    """
    Reference:
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-loss_bce)
        loss_f = self.alpha * (torch.tensor(1.0) - pt) ** self.gamma * loss_bce
        return loss_f.mean()


def dice(preds, targets):
    smooth = 1e-7
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()  # .float()
    union = preds_flat.sum() + targets_flat.sum()  # .float()

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return dice_score


class DiceLoss(nn.Module):
    def __init__(self, softmax=False):
        super(DiceLoss, self).__init__()
        self.softmax = softmax

    def forward(self, logits, targets):
        if self.softmax:
            # softmax channel-wise
            preds = torch.softmax(logits, dim=1)
        else:
            preds = torch.sigmoid(logits)
        return 1.0 - dice(preds, targets)


class BCEDiceLoss(nn.Module):
    """Loss defined as alpha * BCELoss - (1 - alpha) * DiceLoss"""

    def __init__(self, alpha=0.5):
        # TODO check best alpha
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        loss = self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss
        return loss


class BCEDiceLossWithRegLoss(nn.Module):
    def __init__(self, r_reg=0.5):
        super(BCEDiceLossWithRegLoss, self).__init__()
        self.r_reg = r_reg
        self.r_seg = 1.0 - r_reg

        self.reg_loss = nn.BCEWithLogitsLoss()
        self.seg_loss = BCEDiceLoss()

    def forward(self, logits_seg, logits_reg, targets_seg, targets_reg):
        loss_r = self.reg_loss(logits_reg, targets_reg)
        loss_s = self.seg_loss(logits_seg, targets_seg)
        loss = self.r_reg * loss_r + self.r_seg * loss_s
        return loss