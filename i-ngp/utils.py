import torch
import numpy as np


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' 
    generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # TODO: confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' 
    get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    torch.meshgrid(
        torch.linspace(0, W-1, W, device=device), 
        torch.linspace(0, H-1, H, device=device), 
        indexing='ij'
    ) # create grid of all image coords
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}
    if N <= 0:
        inds = torch.arange(H*W, device=device).expand([B, H*W])
    else:
        N = min(N, H*W)
        # sample based on patch-based, error map, or random sampling based on conditions
        if patch_size > 1: # if use patch-based sampling, ignore error_map
            # random sample left-top coords
            # TODO: this impl will lead to less sampling on the image corner pixels
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = torch.meshgrid(
                torch.arange(patch_size, device=device), 
                torch.arange(patch_size, device=device), 
                indexing='ij'
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]
            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten
            inds = inds.expand([B, N])
        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:
            # weighted sample on a low-res grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random noise
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y
            results['inds_coarse'] = inds_coarse # need this when updating error_map
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds

    # create rays based on random samples
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # quaternion operations to transform rays based on cam2world poses
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    # origin of rays
    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    return results


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


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

