import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

import cpp_backend


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


class CompositeRaysInference(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh=1e-2):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        _cpp_backend.composite_rays(n_alive, n_step, T_thresh, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image)
        return tuple()

