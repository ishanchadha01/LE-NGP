import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn


from utils import sample_pdf
from functions import *


class NerfRenderer(nn.Module):
    def __init__(self,
                 grid_size=128,
                 scale=1, # bounding box bounds for xyz/dir 
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 cuda_ray_marching=True,
                 density_threshold=0.01,
                 min_near=0.2 # lower bound for starting point for ray marching from ray origin
                 ):
        super().__init__()
        self.scale = scale
        self.density_scale = density_scale
        # TODO: this is fine for testing, but there should be scale>1 or else only 1 cascade=1 resolution
        self.cascades = int(1 + np.ceil(np.log2(self.scale))) # power of 2 range, eg self.scale=8 gives 4 cascades, [1,2,4,8]
        self.grid_size = grid_size
        self.cuda_ray_marching = cuda_ray_marching
        self.density_threshold = density_threshold
        self.min_near = min_near

        # create axis-aligned bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
        aabb = torch.tensor([-1, 1, -1, 1, -1, 1])
        # buffer saves data but unline parameter (no grad but updated by optimizer), it's not updated by optimizer
        self.register_buffer("aabb", aabb) 

        # additional info for ray marching in cuda, maintained as extra state
        if self.cuda_ray_marching: # TODO: try turning this off if doesnt work
            # create cascade of grids at multiple resolutions
            # voxel grid essentially 128x128x128 at each of cascade resolutions
            density_grid = torch.zeros((self.cascades, self.grid_size ** 3))
            density_bitfield = torch.zeros((self.cascades * self.grid_size**3 // 8), dtype=torch.uint8) # store it more efficiently
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
        temp_grid = -1 * torch.ones(self.density_grid.shape, device=self.density_bitfield.device)
        
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
                        for cascade in range(self.cascades):
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
            for cascade in range(self.cascades):
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
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, is_determinstic=not self.training).detach() # [num_rays, num_step_samples]
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
            xyzs, dirs, deltas, rays = RaymarchingTrainer.apply(
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
                sigmas *= self.density_scale

                CompositeRaysInference(
                    n_alive, 
                    n_step, 
                    rays_alive, 
                    rays_t, 
                    sigmas, 
                    rgbs, 
                    deltas, 
                    weights_sum, 
                    depth, 
                    image, 
                    T_thresh
                )
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
                "output_activation": "None",
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
        x = (x+self.scale)/(2*self.scale) # change range from [-scale, scale] to [0, scale]
        x = self.xyz_encoder(x)
        h = self.sigma_net(x)
        sigmas = TruncExp.apply(h[:, 0])
        geometric_features = h[:, 1:]

        # Compute colors from rgb net
        d = d/torch.norm(d, dim=1, keepdim=True) # normalize direction
        d = self.dir_encoder((d+1)/2) # changes range from [-1, 1] to [0, 1]
        h = torch.cat([d, geometric_features], dim=-1)
        print(sigmas.shape, d.shape, geometric_features.shape)
        print(h.shape)
        h = self.rgb_net(h)
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
        x = (x+self.scale)/(2*self.scale) # change range from [-scale, scale] to [0, scale]
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
                "output_activation": "None",
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
        x = (x+self.scale)/(2*self.scale) # change range from [-scale, scale] to [0, scale]
        x = self.xyz_encoder(x)
        h = self.sigma_net(x)
        sigmas = TruncExp.apply(h[:, 0])
        geometric_features = h[:, 1:]

        # Compute colors from rgb net
        d = d/torch.norm(d, dim=1, keepdim=True) # normalize direction
        d = self.dir_encoder((d+1)/2) # changes range from [-1, 1] to [0, 1]
        h = torch.cat([d, geometric_features], dim=-1)
        h = self.rgb_net(h)
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
        x = (x+self.scale)/(2*self.scale) # change range from [-scale, scale] to [0, scale]
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
    
 