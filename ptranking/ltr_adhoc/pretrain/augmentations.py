import torch
import torch.nn.functional as F

def zeroes(x, aug_percent, device):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly sets some percentage to zero
    """
    aug_x = F.dropout(x.detach().clone(), aug_percent) * (1. - aug_percent)
    return aug_x


def qgswap(x, aug_percent, device):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly swaps some percentage in-qg
    """
    qg_dim = x.shape[1]
    corrupted_indices_cont = torch.rand(x.shape).to(device)

    corrupted_indices_indicator = (corrupted_indices_cont < aug_percent).to(device)
    dim0_target, dim1_target, dim2_target = torch.where(corrupted_indices_indicator)
    dim0_target, dim1_target, dim2_target = dim0_target.to(device), dim1_target.to(device), dim2_target.to(device)
    dim1_source = torch.randint(0, qg_dim, size=dim1_target.shape).to(device)

    aug_x = x.detach().clone().to(device)
    aug_x[dim0_target, dim1_target, dim2_target] = x[dim0_target, dim1_source, dim2_target]

    return aug_x
    