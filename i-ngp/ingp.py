import torch
from torch_ema import ExponentialMovingAverage
import torch.nn.Module as nn
import numpy as np
import tinycudann as tcnn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

import os
from tqdm import tqdm

import _cpp_backend


#TODO
def sample_pdf(bins, weights, n_samples, det=False):
    pass


class TruncExp(Function):
    '''
    Custom function for truncating exp to [-15, 15] for fp32 to ensure numerical stability
    '''
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class Morton3D(Function):
    '''
    Interleave bits of grid using z-curve for easy lookup with divide and conquer octree method, CUDA impl
    Args:
        coords: [num_coords, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...) 
        TODO: check if the coord range is valid! (current 128 is safe)
    Returns:
        indices: [num_coords], int32, in [0, 128^3)
    '''
    @staticmethod
    def forward(ctx, coords):
        if not coords.is_cuda:
            coords = coords.cuda()
        num_coords = coords.shape[0]
        indices = torch.empty(num_coords, dtype=torch.int32, device=coords.device)
        _cpp_backend.morton3D(coords.int(), num_coords, indices)
        return indices


class InverseMorton3D(Function):
    '''
    Invert morton 3d, CUDA impl
    Args:
        indices: [num_coords], int32, in [0, 128^3)
    Returns:
        coords: [num_coords, 3], int32, in [0, 128)
    '''
    @staticmethod
    def forward(ctx, indices):
        if not indices.is_cuda:
            indices = indices.cuda()
        num_coords = indices.shape[0]
        coords = torch.empty(num_coords, 3, dtype=torch.int32, device=indices.device)
        _cpp_backend.invert_morton3D(indices.int(), num_coords, coords)
        return coords


class Packbits(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, thresh, bitfield=None):
        '''
        Pack up the density grid into a bit field to accelerate ray marching, cuda impl
        Args:
            grid: float, [num_cascades, grid_size ** 3], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [num_cascades, grid_size ** 3 / 8]
        '''
        if not grid.is_cuda: 
            grid = grid.cuda()
        grid = grid.contiguous()
        num_cascades = grid.shape[0]
        H3 = grid.shape[1] # grid size cubed
        num_entries = num_cascades * H3 // 8 # num entries across all hashes
        if bitfield is None:
            bitfield = torch.empty(num_entries, dtype=torch.uint8, device=grid.device)
        _cpp_backend.packbits(grid, num_entries, thresh, bitfield)
        return bitfield


class RayIntersection(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' 
        Find near and far bounds of ray intersection with aabb, cuda impl
        Args:
            rays_o: float, [num_rays, 3]
            rays_d: float, [num_rays, 3]
            aabb: float, [6], (xmin, xmax, ymin, ymax, zmin, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [num_rays]
            fars: float, [num_rays]
        '''
        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        rays_o = rays_o.contiguous().view(-1, 3) # make rays_o contiguous in memory, and shape it into (N*B, 3)
        rays_d = rays_d.contiguous().view(-1, 3) # same as above
        num_rays = rays_o.shape[0]
        nears = torch.empty(num_rays, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(num_rays, dtype=rays_o.dtype, device=rays_o.device)
        _cpp_backend.ray_intersection(rays_o, rays_d, aabb, num_rays, min_near, nears, fars)
        return nears, fars


class RaymarchingTrainer(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, scale, density_bitfield, num_cascades, grid_size, nears, fars, dt_gamma=0, max_steps=1024):
        '''
        Generate points by marching rays
        Args:
            rays_o/d: float, [N, 3]
            scale: float, scalar
            density_bitfield: uint8: [num_cascades * grid_size**3 // 8]
            num_cascades
            grid_size
            nears/fars: float, [N]
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize
        Returns:
            xyzs: float, [M, 3], all generated points' coords (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs
            deltas: float, [M, 2], all generated points' deltas (first for RGB, second for Depth)
            rays: int32, [N, 3], all rays (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 2]] --> points belonging to rays[i, 0]
        '''
        if not rays_o.is_cuda: 
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda: 
            rays_d = rays_d.cuda()
        if not density_bitfield.is_cuda: 
            density_bitfield = density_bitfield.cuda()
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()
        num_rays = rays_o.shape[0] # num rays
        max_points = num_rays * max_steps # init max points number in total

        xyzs = torch.zeros(max_points, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(max_points, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(max_points, 2, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(num_rays, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps
        step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        #TODO: could add noise to rays

        _cpp_backend.train_raymarching(
            rays_o, 
            rays_d, 
            density_bitfield, 
            scale, 
            dt_gamma, 
            max_steps, 
            num_rays, 
            num_cascades, 
            grid_size, 
            max_points, 
            nears, 
            fars, 
            xyzs, 
            dirs, 
            deltas, 
            rays, 
            step_counter
        )
        #TODO could keep counter and empty cuda cache for first few epochs and initialize with mean value
        return xyzs, dirs, deltas, rays


class CompositeRayTrainer(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [max_points, 3]
            sigmas: float, [max_points,]
            deltas: float, [max_points, 2]
            rays: int32, [max_points, 3]
        Returns:
            weights_sum: float, [num_rays,], the alpha channel
            depth: float, [num_rays, ], the Depth
            image: float, [num_rays, 3], the RGB channel (after multiplying alpha!)
        '''
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        max_points = sigmas.shape[0]
        num_rays = rays.shape[0]
        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)
        _cpp_backend.train_composite_rays_forward(sigmas, rgbs, deltas, rays, max_points, num_rays, T_thresh, weights_sum, depth, image)
        ctx.save_for_backward(sigmas, rgbs, deltas, rays, weights_sum, depth, image)
        ctx.dims = [max_points, num_rays, T_thresh]
        return weights_sum, depth, image

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights_sum, grad_depth, grad_image):
        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()
        sigmas, rgbs, deltas, rays, weights_sum, depth, image = ctx.saved_tensors
        max_points, num_rays, T_thresh = ctx.dims
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)
        _cpp_backend.train_composite_rays_backward(grad_weights_sum, grad_image, sigmas, rgbs, deltas, rays, weights_sum, image, max_points, num_rays, T_thresh, grad_sigmas, grad_rgbs)
        return grad_sigmas, grad_rgbs, None, None, None


class MarchRaysInference(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, scale, density_bitfield, num_cascades, grid_size, near, far, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            scale: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            num_cascades: int
            grid_size: int
            nears/fars: float, [N]
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        '''
        if not rays_o.is_cuda: 
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda: 
            rays_d = rays_d.cuda()
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        max_points = n_alive * n_step
        xyzs = torch.zeros(max_points, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(max_points, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(max_points, 2, dtype=rays_o.dtype, device=rays_o.device) # 2 vals, one for rgb, one for depth
        _cpp_backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, scale, dt_gamma, max_steps, num_cascades, grid_size, density_bitfield, near, far, xyzs, dirs, deltas)
        return xyzs, dirs, deltas

#TODO
class CompositeRaysInference(Function):
    pass


class NerfRenderer(nn):
    def __init__(self,
                 grid_size=128,
                 scale=1, # bounding box bounds for xyz/dir 
                 cuda_ray_marching=True,
                 density_threshold=0.01,
                 min_near=0.2 # lower bound for starting point for ray marching from ray origin
                 ):
        super().__init__()
        self.scale = scale
        # TODO: this is fine for testing, but there should be scale>1 or else only 1 cascade=1 resolution
        self.cascades = 1 + np.ceil(np.log2(self.scale)) # power of 2 range, eg self.scale=8 gives 4 cascades, [1,2,4,8]
        self.grid_size = grid_size
        self.cuda_ray_marching = cuda_ray_marching
        self.density_threshold = density_threshold

        # create axis-aligned bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
        self.aabb = torch.tensor([-1, 1, -1, 1, -1, 1])
        # buffer saves data but unline parameter (no grad but updated by optimizer), it's not updated by optimizer
        self.register_buffer("aabb", self.aabb) 

        # additional info for ray marching in cuda, maintained as extra state
        if self.cuda_ray_marching: # TODO: try turning this off if doesnt work
            # create cascade of grids at multiple resolutions
            # voxel grid essentially 128x128x128 at each of cascade resolutions
            density_grid = torch.zeros((self.cascades, self.grid_size ** 3))
            density_bitfield = torch.zeros((self.cascades * self.grid_size**3 // 8)) # store it more efficiently
            self.register_buffer("density_grid", density_grid)
            self.register_buffer("density_bitfield", density_bitfield)
            self.num_iters_density_update = 0 # perform full update of density if its still first couple iters

    def forward(self, *args):
        raise NotImplementedError()
    
    def density(self, *args):
        raise NotImplementedError()
    

    def reset_extra_state(self):
        self.density_grid.zero_()
        self.num_iters_density_update = 0


    @torch.no_grad()
    def update_extra_state(self, decay=0.95, num_splits=128):
        if not self.cuda_ray_marching:
            return
        # update density grid
        temp_grid = -1 * torch.ones(self.density_grid.shape)
        
        # if less than 16 iters have passed, then do full update, otherwise do partial update
        if self.num_iters_density_update < 16:
            # split grid into num_splits tensors
            x_vals = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(num_splits)
            y_vals = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(num_splits)
            z_vals = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(num_splits)

            for x_split in x_vals:
                for y_split in y_vals:
                    for z_split in z_vals:
                        # construct octree binary representation with morton encoding
                        # each tensor in meshgrid is values of i^th dim arranged so that all combos of coords can be made w/ n^th dim
                        xx, yy, zz = torch.meshgrid(x_split, y_split, z_split, indexing='ij') # requires torch>=2
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = Morton3D.apply(coords).long() # [num_coords] interleave values of 3 fields for raymarching indices
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [num_coords, 3], change range of xyz to [-1, 1]

                        # cascading at multiple resolutions
                        for cascade in self.cascades:
                            scale = min(self.scale, 2**cascade)
                            half_grid_size = self.scale / self.grid_size
                            cascade_xyzs = xyzs * (scale - half_grid_size) # scale to current cascades resolution
                            cascade_xyzs += (torch.rand_like(cascade_xyzs) * 2 - 1) * half_grid_size # add noise in [-half grid size, half grid size]

                            # query density net to get depths
                            sigmas = self.density(cascade_xyzs)['sigmas'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            temp_grid[cascade, indices] = sigmas # assign in new hash grid[cascade] for this resolution

        else:
            # partial update
            num_coords = self.grid_size ** 3 // 4 # grid_size**3 / 4
            for cascade in self.cascades:
                # randomly sample few positions
                coords = torch.randint(0, self.grid_size, (num_coords, 3), device=self.density_bitfield.device) # [num_coords, 3], in [0, 128)
                indices = Morton3D.apply(coords).long() # [num_coords]
                # randomly sample occupied positions
                occupied_indices = torch.nonzero(self.density_grid[cascade] > 0).squeeze(-1) # [num_coords * z]
                rand_mask = torch.randint(0, occupied_indices.shape[0], [num_coords], dtype=torch.long, device=self.density_bitfield.device)
                occupied_indices = occupied_indices[rand_mask] # [num_coords * z] --> [num_coords], allow for duplication
                occupied_coords = InverseMorton3D.apply(occupied_indices) # [num_coords, 3], decompressed from octree morton encoding
                # concatenate, allowing for duplication with previous random sample since it has both occupied/unoccupied
                indices = torch.cat([indices, occupied_indices], dim=0)
                coords = torch.cat([coords, occupied_coords], dim=0)

                # below is same as full update
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [num_coords, 3], change range of xyz to [-1, 1]
                scale = min(self.scale, 2**cascade)
                half_grid_size = self.scale / self.grid_size
                cascade_xyzs = xyzs * (scale - half_grid_size)
                cascade_xyzs += (torch.rand_like(cascade_xyzs) * 2 - 1) * half_grid_size
                sigmas = self.density(cascade_xyzs)['sigmas'].reshape(-1).detach()
                sigmas *= self.density_scale
                temp_grid[cascade, indices] = sigmas

        # ema update, select max between decayed current grid and new temp grid
        valid_mask = (self.density_grid >= 0) & (temp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, temp_grid[valid_mask])
        self.num_iters_density_update += 1

        # convert to bitfield
        # TODO could take min of mean density thresh and self.density thresh here
        self.density_bitfield = Packbits.apply(self.density_grid, self.density_threshold, self.density_bitfield)

        #TODO: could add stepcounters here for mean, local/global step


    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, num_splits=64):
        # poses: [num_batches, 4, 4]
        # intrinsic: [3, 3]
        # Marks untrained regions of grid with -1
        if not self.cuda_ray_marching:
            return
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        num_batches = poses.shape[0]
        fx, fy, cx, cy = intrinsic # focal lengths and camera center, but this is 3x3 so why does unwrapping like this work?

        # split grid into num_splits tensors
        x_vals = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(num_splits)
        y_vals = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(num_splits)
        z_vals = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(num_splits)

        counts = torch.zeros_like(self.density_grid)
        poses = poses.to(counts.device)

        for x_split in x_vals:
            for y_split in y_vals:
                for z_split in z_vals:
                    # construct octree binary representation with morton encoding
                    # each tensor in meshgrid is values of i^th dim arranged so that all combos of coords can be made w/ n^th dim
                    xx, yy, zz = torch.meshgrid(x_split, y_split, z_split, indexing='ij') # requires torch>=2
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = Morton3D.apply(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cascade in range(self.cascades):
                        scale = min(2 ** cascade, self.scale)
                        half_grid_size = scale / self.grid_size
                        cascade_world_xyzs = world_xyzs * (scale - half_grid_size)

                        # split batch to avoid running out of mem
                        head = 0
                        while head < num_batches:
                            tail = min(head + num_splits, num_batches)

                            #TODO:
                            # world2cam transform
                            # poses is c2w, so we need to transpose it
                            # another transpose is needed for batched matmul, so the final form is without transpose
                            cam_xyzs = cascade_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [num_splits, num_coords, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [num_splits, num_coords]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [num_coords]

                            # update count 
                            counts[cascade, indices] += mask
                            head += num_splits
    
        # mark untrained grid as -1
        self.density_grid[counts == 0] = -1


    def render(self, rays_o, rays_d):
        # rays_o, rays_d: [num_batches, num_rays, 3]
        # return: rgb_prds: [num_batches, num_rays, 3]
        if self.cuda_ray_marching:
            return self.run_cuda(rays_o, rays_d)
        else:
            return self.run(rays_o, rays_d) #TODO could also used staged computation if without cuda


    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128):
        # rays_o, rays_d: [num_batches, num_rays, 3], assume num_batches is 1 TODO why?
        # bg_color: [3] in range [0, 1]
        # return: image: [num_batches, num_rays, 3], depth: [num_batches, num_rays]
        prefix = rays_o.shape[:-1] # image dims
        rays_o = rays_o.contiguous().view(-1, 3) # make rays_o contiguous in memory, and shape it into (N*B, 3)
        rays_d = rays_d.contiguous().view(-1, 3) # same as above
        num_rays = rays_o.shape[0] # num_rays = num_batches * num_rays since B=1
        device = rays_o.device
        
        # near distance is where ray marching starts, far is where it decides to terminate
        nears, fars = RayIntersection.apply(rays_o, rays_d, self.aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # collect num_steps samples
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, num_steps]
        z_vals = z_vals.expand((num_rays, num_steps)) # [num_rays, num_steps]
        z_vals = nears + (fars - nears) * z_vals # [num_rays, num_steps], in [nears, fars]

        # TODO could add noise to z vals

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [num_rays, 1, 3] + [num_rays, 1, 3] * [num_rays, num_steps, 1] -> [num_rays, num_steps, 3] with broadcasting
        xyzs = torch.min(torch.max(xyzs, self.aabb[:3]), self.aabb[3:]) # manually clip

        # TODO for debugging, could plot pointcloud here
        density_outputs = self.density(xyzs.reshape(-1, 3))

        # since we get sigmas and geometric features
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(num_rays, num_steps, -1) # view is essentially in place reshape
        
        # upsample, like in nerf
        sample_dist = (fars - nears) / num_steps
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [num_rays, num_steps-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1) # elementwise difference in sampled distances along rays
                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [num_rays, num_steps], exponential weighting assigment to reduce importance of far away weights
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [num_rays, num_steps+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [num_rays, num_steps], final weighting assignment

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [num_rays, num_steps-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [num_rays, num_step_samples]
                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [num_rays, 1, 3] + [num_rays, 1, 3] * [num_rays, num_step_samples, 1] -> [num_rays, num_step_samples, 3], with broadcasting
                new_xyzs = torch.min(torch.max(new_xyzs, self.aabb[:3]), self.aabb[3:]) # manually clip

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(num_rays, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [num_rays, sampled_z_vals+z_vals (same as num_steps+sampled_num_steps)]
            z_vals, z_index = torch.sort(z_vals, dim=1)
            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [num_rays, sampled_z_vals+z_vals, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))
            for k in density_outputs:
                temp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(temp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(temp_output))
            # end upsampling

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [num_rays, sampled_z_vals+z_vals-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [num_rays, sampled_z_vals+z_vals]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [num_rays, sampled_z_vals+z_vals+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [num_rays, sampled_z_vals+z_vals]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4 # TODO hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3) # [num_rays, sampled_z_vals+z_vals, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [num_rays]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1) # normalized version of original z samples
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        bg_color = 1 #TODO: could randomize background color, this is done in paper!

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        return {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
        }


    #TODO
    def run_cuda(self, rays_o, rays_d, max_steps=1024, dt_gamma=0, T_thresh=1e-4):
        # rays_o, rays_d: [num_batches, num_rays, 3], assume num_batches is 1 TODO why?
        # bg_color: [3] in range [0, 1]
        # dt_gamma: cone angle
        # return: image: [num_batches, num_rays, 3], depth: [num_batches, num_rays]
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        num_rays = rays_o.shape[0] # num_rays = num_batches * num_rays since B=1
        device = rays_o.device
        
        # near distance is where ray marching starts, far is where it decides to terminate
        nears, fars = RayIntersection.apply(rays_o, rays_d, self.aabb, self.min_near)

        bg_color = 1 #TODO: could randomize background color, this is done in paper!

        results = {}
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0

            xyzs, dirs, deltas, rays = RaymarchingTrainer(
                rays_o, 
                rays_d, 
                scale=self.scale, 
                density_bitfield=self.density_bitfield, 
                num_cascades=self.cascades, 
                grid_size=self.grid_size, 
                nears=nears, 
                fars=fars,
                dt_gamma=dt_gamma, 
                max_steps=max_steps
            )

            sigmas, rgbs = self(xyzs, dirs)
            sigmas = self.density_scale * sigmas

            weights_sum, depth, image = CompositeRayTrainer(sigmas, rgbs, deltas, rays, T_thresh)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            
            results['weights_sum'] = weights_sum

        else:
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(num_rays, dtype=dtype, device=device)
            depth = torch.zeros(num_rays, dtype=dtype, device=device)
            image = torch.zeros(num_rays, 3, dtype=dtype, device=device)
            
            n_alive = num_rays
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [num_rays]
            rays_t = nears.clone() # [num_rays]

            step = 0
            while step < max_steps:
                # count alive rays 
                n_alive = rays_alive.shape[0]
                if n_alive == 0: # break if no alive rays
                    break

                # decide compact_steps
                n_step = max(min(num_rays // n_alive, 8), 1)

                #TODO: march rays function!!!
                xyzs, dirs, deltas = MarchRaysInference(
                    n_alive, 
                    n_step, 
                    rays_alive, 
                    rays_t, 
                    rays_o, 
                    rays_d, 
                    self.scale, 
                    self.density_bitfield, 
                    self.cascades, 
                    self.grid_size, 
                    nears, 
                    fars, 
                    dt_gamma, 
                    max_steps
                )
                sigmas, rgbs = self(xyzs, dirs)
                sigmas = self.density_scale * sigmas

                # TODO: composite rays function
                CompositeRaysInference(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
                rays_alive = rays_alive[rays_alive >= 0]
                step += n_step
        
            # Construct image after inference loop done
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image
        return results


class INGP(NerfRenderer):
    def __init__(self,
                 geometric_features_dim=15, # first feature is sigma, all the rest are geometric features
                 num_layers=2,
                 hidden_dim=64,
                 scale=1):
        super().__init__()

        self.scale = scale
        self.num_layers = num_layers
        self.geometric_features_dim = geometric_features_dim
        self.hidden_dim = hidden_dim

        L = 16; F = 2; log2_T = 19; N_min = 16
        per_level_scale = np.exp(np.log(2048*self.scale/N_min)/(L-1))

        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_T,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32, # 16 for dir encoding, 16 for geometric features
            n_output_dims=1+self.geometric_features_dim, # 1 for sigmas, 15 for geometric features
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers,
            },
        )

        self.dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=32, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": self.rgb_act,
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers,
            }
        )

        # TODO: some implementations have a toning network after too if no rgb activation

    def forward(self, x, d):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """

        # Compute densities from sigma net
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min) # change range from [-scale, scale] to [0, scale]
        x = self.xyz_encoder(x)
        h = self.sigma_net(x)
        sigmas = TruncExp.apply(h[:, 0])
        geometric_features = h[:, 1:]

        # Compute colors from rgb net
        d = d/torch.norm(d, dim=1, keepdim=True) # normalize direction
        d = self.dir_encoder((d+1)/2) # changes range from [-1, 1] to [0, 1]
        h = torch.cat([d, geometric_features], dim=-1)
        h = self.color_net(h)
        rgbs = torch.sigmoid(h) # sigmoid activation for rgb

        return sigmas, rgbs


    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]

        Outputs:
            sigmas: (N), separated into sigmas and geometric features
        """

        # Compute densities from sigma net
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min) # change range from [-scale, scale] to [0, scale]
        x = self.xyz_encoder(x)
        h = self.sigma_net(x)
        sigmas = TruncExp.apply(h[:, 0])
        geometric_features = h[:, 1:]
        return {
            "sigmas": sigmas,
            "geometric_features": geometric_features
        }


    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.xyz_encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.dir_encoder.parameters(), 'lr': lr},
            {'params': self.light_encoder.parameters(), 'lr': lr},
            {'params': self.rgb_net.parameters(), 'lr': lr}, 
        ]
        
        return params
    

#TODO
class LENGP(NerfRenderer):
    def __init__(self,
                 geometric_features_dim=15, # first feature is sigma, all the rest are geometric features
                 num_layers=2,
                 hidden_dim=64,
                 scale=1):
        super().__init__()

        self.scale = scale
        self.num_layers = num_layers
        self.geometric_features_dim = geometric_features_dim
        self.hidden_dim = hidden_dim

        L = 16; F = 2; log2_T = 19; N_min = 16
        per_level_scale = np.exp(np.log(2048*self.scale/N_min)/(L-1))

        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_T,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32, # 16 for dir encoding, 16 for geometric features
            n_output_dims=1+self.geometric_features_dim, # 1 for sigmas, 15 for geometric features
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers,
            },
        )

        self.dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        #TODO: adjust this after trying it
        self.light_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=32, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": self.rgb_act,
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers,
            }
        )

        # TODO: some implementations have a toning network after too if no rgb activation

    def forward(self, x, d, l):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
            l: (N, 3) light position

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        #TODO edit this to include light

        # Compute densities from sigma net
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min) # change range from [-scale, scale] to [0, scale]
        x = self.xyz_encoder(x)
        h = self.sigma_net(x)
        sigmas = TruncExp.apply(h[:, 0])
        geometric_features = h[:, 1:]

        # Compute colors from rgb net
        d = d/torch.norm(d, dim=1, keepdim=True) # normalize direction
        d = self.dir_encoder((d+1)/2) # changes range from [-1, 1] to [0, 1]
        h = torch.cat([d, geometric_features], dim=-1)
        h = self.color_net(h)
        rgbs = torch.sigmoid(h) # sigmoid activation for rgb

        return sigmas, rgbs
    

    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]

        Outputs:
            sigmas: (N), separated into sigmas and geometric features
        """

        # Compute densities from sigma net
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min) # change range from [-scale, scale] to [0, scale]
        x = self.xyz_encoder(x)
        h = self.sigma_net(x)
        sigmas = TruncExp.apply(h[:, 0])
        geometric_features = h[:, 1:]
        return {
            "sigmas": sigmas,
            "geometric_features": geometric_features
        }


    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.xyz_encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.dir_encoder.parameters(), 'lr': lr},
            {'params': self.light_encoder.parameters(), 'lr': lr},
            {'params': self.rgb_net.parameters(), 'lr': lr}, 
        ]
        
        return params
    
    

#TODO
class NerfDataset():
    pass


#TODO possibly LPIPS and others too
class PSNRMeter():
    def __init__(self):
        self.val = 0
        self.num = 0

    def update(self, pred, gt):
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        mse += np.mean(np.square(pred-gt))
        psnr =  -10*np.log(mse) # pixel max is 1 so psnr simplifies
        self.val += psnr
        self.num += 1

    def measure(self):
        return self.val / self.num
    
    def write(self):
        # unclear if this is needed
        pass

    def report(self):
        return f"PSNR = {self.measure:.6f}"
    
    def clear(self):
        self.val = 0
        self.num = 0
        


#TODO
class Trainer():
    def __init__(self, 
                 model,
                 lr_scheduler, 
                 criterion,
                 optimizer,
                 device,
                 experiment_name="ngp",
                 workspace="./workspace", 
                 ema_decay=0.95, # for smoothing
                 fp16=True, 
                 scheduler_update_every_step=True, 
                 metrics=[PSNRMeter()], 
                 use_checkpoint="latest", 
                 eval_interval=50,
                 local_rank=0 # device id if doing distributed training
                 ):

        self.experiment_name = experiment_name
        self.optimizer = optimizer
        self.device = device
        self.workspace = workspace
        self.criterion = criterion
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.lr_scheduler = lr_scheduler
        self.scheduler_update_every_step = scheduler_update_every_step
        self.metrics = metrics
        self.use_checkpoint = use_checkpoint
        self.eval_interval = eval_interval
        self.local_rank - local_rank

        #TODO: could do distributed training
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model.to(self.device)
        if isinstance(criterion, nn):
            criterion.to(device)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.model)
        self.lr_scheduler = lr_scheduler(self.optimizer)
        self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16) # speed up computation with amp
        
        # TODO: could import lpips for patch based training for large scenes

        # initialize other variables
        self.epoch = 0
        self.local_step = 0
        self.global_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # prepare workspace
        os.makedirs(self.workspace, exist_ok=True)
        self.log_path = os.path.join(workspace, f"log_{self.experiment_name}.txt")
        self.log_ptr = open(self.log_path, "a+")
        self.ckpt_path = os.path.join(self.workspace, "checkpoints")
        self.best_path - os.path.join(self.ckpt_path, f"{experiment_name}.pth")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.load_checkpoint(self.use_checkpoint) # can configure this to be latest, best, etc

        # can use a clip based loss? why would an image to text classifier provide good loss function for this?
        # TODO: maybe similar to text guided object detection like dreamfields

    def __del__(self):
        self.log_ptr.close()

    def log(self, *args):
        if self.local_rank==0 and self.log_ptr: # only log on local device if log_ptr exists
            print(*args, file=self.log_ptr)
            self.log_ptr.flush() # write to file immediately

    def train_step(self, data):
        # trace rays through image to render them
        rays_o = data["rays_o"]
        rays_d = data["rays_d"]

    def train_one_epoch(self, train_loader):
        self.log(f"Training Epoch {self.epoch}")
        self.model.train() # Put model in training mode

        # Clear metrics
        for metric in self.metrics:
            metric.clear()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        train_loader.sampler.set_epoch(self.epoch) #TODO: might not need to do this due to small world size
        pbar = tqdm.tqdm(
            total=len(train_loader) * train_loader.batch_size, 
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for data in train_loader:

            # render it first??

            with torch.cuda.amp.autocast_mode():
                self.train_step()

    def eval_one_epoch(self):
        pass

    def eval_step(self):
        pass

    def test_step(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def test(self):
        pass

    def save_ckpt(self):
        pass

    def load_ckpt(self):
        pass

    def save_mesh(self):
        pass







def main():
    model = INGP()
    print(model)
    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda')
    lr = 1e-3
    iters = 1e4

    # Training
    optimizer = lambda model: torch.optim.Adam(model.get_params(lr), betas=(0.9, 0.99), eps=1e-15)
    train_loader = NerfDataset(device=device, type='train').dataloader()

    # decay to 0.1 * init_lr at last iter step
    scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / iters, 1))
    metrics = [PSNRMeter()]
    trainer = Trainer('ngp', 
                      model, 
                      device=device, 
                      optimizer=optimizer,
                      criterion=criterion, 
                      lr_scheduler=scheduler, 
                      metrics=metrics)
    valid_loader = NerfDataset(device=device, type='val', downscale=1).dataloader()

    max_epoch = np.ceil(iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, max_epoch)

    # also test
    test_loader = NerfDataset(device=device, type='test').dataloader()
    
    if test_loader.has_gt:
        trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
    trainer.test(test_loader, write_video=True) # test and save video
    
    trainer.save_mesh(resolution=256, threshold=10)

    # Testing
    metrics = [PSNRMeter()]
    trainer = Trainer(model, device=device, criterion=criterion, metrics=metrics)

    
    test_loader = NerfDataset(device=device, type='test').dataloader()

    if test_loader.has_gt: # check for ground truth
        trainer.evaluate(test_loader) # blender has gt, so evaluate it.

    trainer.test(test_loader, write_video=True) # test and save video
    
    trainer.save_mesh(resolution=256, threshold=10)