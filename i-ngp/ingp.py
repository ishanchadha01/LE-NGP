import torch
from torch_ema import ExponentialMovingAverage
import torch.nn.Module as nn
import numpy as np
import tinycudann as tcnn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
import os
from tqdm import tqdm


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



class NerfRenderer(nn):
    def __init__(self,
                 grid_size=128,
                 scale=1, # bounding box bounds for xyz/dir 
                 cuda_ray_marching=True,
                 ):
        self.scale = scale
        # TODO: this is fine for testing, but there should be scale>2 or else only 1 cascade=1 resolution
        self.cascade = 1 + np.ceil(np.log2(self.scale))
        self.grid_size = grid_size
        self.cuda_ray_marching = cuda_ray_marching

        # create axis-aligned bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
        self.aabb = torch.tensor([-1, 1, -1, 1, -1, 1])
        # buffer saves data but unline parameter (no grad but updated by optimizer), it's not updated by optimizer
        self.register_buffer("aabb", self.aabb) 

        # extra state(?) for ray marching in cuda
        if self.cuda_ray_marching: # TODO: try turning this off if doesnt work
            # create cascade of grids at multiple resolutions
            # voxel grid essentially 128x128x128 at each of cascade resolutions
            density_grid = torch.zeros((self.cascades, self.grid_size ** 3))
            density_bitfield = torch.zeros((self.cascades * self.grid_size**3 // 8)) # store it more efficiently
            self.register_buffer("density_grid", density_grid)
            self.register_buffer("density_bitfield", density_bitfield)

    def forward(self, **args):
        raise NotImplementedError()
    
    #TODO
    def reset_extra_state(self):
        pass

    #TODO
    @torch.no_grad()
    def update_extra_state(self):
        pass

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
        sigmas = TruncExp.apply(sigmas[:, 0])
        geometric_features = sigmas[:, 1:]

        # Compute colors from rgb net
        d = d/torch.norm(d, dim=1, keepdim=True) # normalize direction
        d = self.dir_encoder((d+1)/2) # changes range from [-1, 1] to [0, 1]
        h = torch.cat([d, geometric_features], dim=-1)
        h = self.color_net(h)
        rgbs = torch.sigmoid(h) # sigmoid activation for rgb

        return sigmas, rgbs

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
        sigmas = TruncExp.apply(sigmas[:, 0])
        geometric_features = sigmas[:, 1:]

        # Compute colors from rgb net
        d = d/torch.norm(d, dim=1, keepdim=True) # normalize direction
        d = self.dir_encoder((d+1)/2) # changes range from [-1, 1] to [0, 1]
        h = torch.cat([d, geometric_features], dim=-1)
        h = self.color_net(h)
        rgbs = torch.sigmoid(h) # sigmoid activation for rgb

        return sigmas, rgbs

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