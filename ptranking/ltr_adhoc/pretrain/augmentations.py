import torch
import numpy as np
import torch.nn.functional as F

# def zeroes(x, aug_percent, device, categorical_features, mix=0., scale=0.):
#     """
#     Takes x of dimension [batch, num_docs, num_features], and randomly sets some percentage to zero
#     """
#     probability_mask = torch.bernoulli(torch.full(x.shape, 1 - aug_percent)).to(device)
#     for idx in categorical_features.keys():
#         probability_mask[:, :, idx] = 1
#     zeroed_matrix = x.detach().clone() * probability_mask

#     return zeroed_matrix

def dacl(x, aug_percent, device, mix=0., scale=0.):
    num_features = x.shape[2]
    orig_shape = x.shape
    
    feature_bank = x.detach().clone().to(device).reshape(-1, num_features)
    num_samples_batch = feature_bank.shape[0]
    x_full = x.reshape(-1, num_features)
    randidx = torch.multinomial(torch.ones(num_samples_batch), num_samples=num_samples_batch, replacement=True)
    sampled = feature_bank[randidx,:]

    random_mixup = torch.rand(1)

    random_mixup_weight = (torch.rand(1) * aug_percent).to(device)
    mask_percent = aug_percent/2.0
    if random_mixup < 1./2.:
        res = (1.0 - random_mixup_weight) * x_full + (random_mixup_weight) * (sampled)
    else:
        mask = torch.bernoulli(torch.ones_like(x_full).to(device) * mask_percent).to(device)
        res = (1 - mask) * x_full + (mask) * sampled
    returned_result = res.reshape(orig_shape)
    return returned_result

def scarf(x, aug_percent, device, mix=0., scale=0.):
    num_features = x.shape[2]
    orig_shape = x.shape
    x_full = x.reshape(-1, num_features)

    corrupted_indices_cont = torch.rand(x_full.shape).to(device)
    corrupted_indices_indicator = (corrupted_indices_cont < aug_percent).to(device)
    dim0_target, dim1_target = torch.where(corrupted_indices_indicator)
    dim0_source = torch.randint(0, x_full.shape[0], size=dim0_target.shape).to(device)
    aug_x = x_full.detach().clone().to(device)
    aug_x[dim0_target, dim1_target] = x_full[dim0_source, dim1_target].detach().clone().to(device)
    
    return aug_x.reshape(orig_shape)

def qgswap(x, aug_percent, device, mix=0., scale=0.):
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
    aug_x[dim0_target, dim1_target, dim2_target] = x[dim0_target, dim1_source, dim2_target].detach().clone().to(device)

    return aug_x
def gaussian(x, aug_percent, device):
    aug_x = x.detach().clone().to(device)
    noise_aug_x = aug_x + torch.randn_like(aug_x, device=device) * aug_percent
    return noise_aug_x
def zeroes(x, aug_percent, device, mix=0., scale=0.):
    """
    Takes x of dimension [batch, num_docs, num_features], and randomly sets some percentage to zero
    """
    aug_x = F.dropout(x.detach().clone(), aug_percent) * (1. - aug_percent)
    noise_aug_x = aug_x + torch.randn_like(aug_x, device=device) * scale
    return noise_aug_x    
# def gaussian(x, aug_percent, device, categorical_features):
#     feature_dim = x.size(2)  # Assuming the last dimension is the feature dimension
#     mask = torch.ones(feature_dim, dtype=torch.bool).to(device)  # Start with a mask that includes all features
#     for idx in categorical_features.keys():
#         mask[idx] = False
#     mask = mask.unsqueeze(0).unsqueeze(0)  # Reshape mask to broadcast over batch and qg_size dimensions
#     mask = mask.expand_as(x)  # Ensure the mask has the same shape as the matrix
#     noise = torch.randn_like(x, device=device) * aug_percent
#     noisy_x = x.detach().clone() + noise * mask.float()
#     return noisy_x

def categorical_augment(x, aug_percent, device, categorical_features):
    if aug_percent >= 1.0:
        scale_to_p = {1.0: 0.2, 1.5: 0.4, 2.0: 0.6, 2.5: 0.7, 3.0: 0.9}
        p = scale_to_p[aug_percent]
    else:
        p = aug_percent
    batch_size, qg_size, feature_dim = x.shape
    categorical_mask = torch.zeros(batch_size, qg_size, feature_dim).to(device)

    # Only target the categorical features for modification
    for idx in categorical_features.keys():
        categorical_mask[:, :, idx] = 1

    # Determine which categorical features to modify based on probability p
    modification_mask = torch.bernoulli(categorical_mask * p).to(device)

    # Step 2: Generate random values (-1 or 1) for each categorical feature
    # Map Bernoulli(0.5) outcomes (0 or 1) to -1 or 1
    random_values = torch.bernoulli(torch.full(x.shape, 0.5)).to(device) * 2 - 1

    # Apply random values only to the categorical features designated for modification
    categorical_changes = random_values * modification_mask

    # Step 3: Apply the changes to the matrix
    # Ensure that only the categorical features designated for modification are affected
    modified_matrix = x.detach().clone()  # Create a copy to modify
    modified_matrix[categorical_mask.bool()] = 0  # Zero out the original categorical feature values
    modified_matrix += categorical_changes  # Apply the new -1 or 1 values

    return modified_matrix

def qg_and_zero(x, aug_percent, device, mix=0., scale=0.):
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
    candidate_replacements = x[dim0_target, dim1_source, dim2_target].to(device)
    random_zero_percent = mix
    candidate_replacements_zero = F.dropout(candidate_replacements, random_zero_percent) * (1. - random_zero_percent)
    aug_x[dim0_target, dim1_target, dim2_target] = candidate_replacements_zero
    noise_aug_x = aug_x + torch.randn_like(aug_x, device=device) * scale
    return noise_aug_x    
