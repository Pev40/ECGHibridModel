import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # logits, targets: [B, C]
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # focal modulation
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            loss_pos *= ((1 - xs_pos) ** self.gamma_pos)
            loss_neg *= (xs_pos ** self.gamma_neg)

        loss = - (loss_pos + loss_neg)
        return loss.mean()


