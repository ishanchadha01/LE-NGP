import torch


def sample_pdf(bins, weights, n_samples, is_deterministic=False):
    # bins: [num_batches, num_z], old_z_vals
    # weights: [num_batches, num_z - 1], bin weights.
    # return: [num_batches, n_samples], new_z_vals
    # use pdf to get cdf, use cdf to take uniform random samples, invert cdf to generate new samples based on cdf

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) # along last dim bc 1st dim is batches
    cdf = torch.cumsum(pdf, -1) # along last dim bc 1st dim is batches
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) # concat zeros to the front to make sure cdf starts at 0

    # Take uniform samples
    if is_deterministic: # generate deterministic samples if in testing phase
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous() # ensures samples are contiguous in mem for faster mem access
    inds = torch.searchsorted(cdf, u, right=True) # uses binary search to find where samples should be inserted into cdf
    below = torch.max(torch.zeros_like(inds - 1), inds - 1) # idx below insertion point, ensuring >=0
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds) # idx above insertion point, ensuring <len
    inds_g = torch.stack([below, above], -1)  # (num_batches, n_samples, 2), stack below and above along last dim
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # gets vals from expanded cdf based on inds_g
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # same as above for bins
    denom = (cdf_g[..., 1] - cdf_g[..., 0]) # difference between upper and lower bounds of bins
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom) # clamped so denom>0
    t = (u - cdf_g[..., 0]) / denom # normalize diff between u and lower bound of computed denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]) # compute samples using linear interpolation

    return samples

