import torch
import torch.nn.functional as F

def zeroes(x, aug_percent):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly sets some percentage to zero
    """
    aug_x = F.dropout(x.detach().copy(), aug_percent) * (1. - aug_percent)
    return aug_x
