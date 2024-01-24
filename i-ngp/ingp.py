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
        if not coords.is_cuda():
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
        if not indices.is_cuda():
            indices = indices.cuda()
        num_coords = indices.shape[0]
        coords = torch.empty(num_coords, 3, dtype=torch.int32, device=indices.device)
        _cpp_backend.invert_morton3D(indices.int(), num_coords, coords)
        return coords


class NerfRenderer(nn):
    def __init__(self,
                 grid_size=128,
                 scale=1, # bounding box bounds for xyz/dir 
                 cuda_ray_marching=True,
                 ):
        super().__init__()
        self.scale = scale
        # TODO: this is fine for testing, but there should be scale>1 or else only 1 cascade=1 resolution
        self.cascades = 1 + np.ceil(np.log2(self.scale)) # power of 2 range, eg self.scale=8 gives 4 cascades, [1,2,4,8]
        self.grid_size = grid_size
        self.cuda_ray_marching = cuda_ray_marching

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
    
    #TODO
    def reset_extra_state(self):
        self.density_grid.zero_()
        self.num_iters_density_update = 0

    #TODO
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


    #TODO
    @torch.no_grad()
    def mark_untrained_grid(self):
        pass

    #TODO
    def render(self):
        pass

    #TODO
    def run(self):
        pass

    #TODO
    def run_cuda(self):
        pass


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