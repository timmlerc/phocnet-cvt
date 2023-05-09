import torch
from torch import abs, sigmoid, log, sum, mean, clamp


class WeightedBinaryCrossEntropyLoss(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction == 'mean':
            self.reduction = mean
        elif reduction == 'sum':
            self.reduction = sum
        else:
            raise ValueError('Unsupported reduction method {:s}'.format(reduction))

    def forward(self, estimates, target, exponents=1., term_weights=1.):

        estimates = sigmoid(estimates)
        estimates = clamp(estimates, min=1e-15, max=1.-1e-15)

        #p_weights = (1. - estimates).pow(exponents)
        #q_weights = estimates.pow(exponents)

        p_weights = abs(1. - estimates)
        q_weights = abs(estimates)

        p = log(estimates)
        q = log(1. - estimates)

        p_cross_entropy = target * p * p_weights
        q_cross_entropy = (1. - target) * q * q_weights

        loss = p_cross_entropy + q_cross_entropy
        loss = self.reduction(-loss * term_weights)
        return loss
