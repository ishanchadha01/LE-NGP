import torch
import torch.nn.Module as nn
import numpy as np
import tinycudann as tcnn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 


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
    

class INGP(nn):
    def __init__(self,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 scale=1,):
        super().__init__()

        self.scale = scale

        L = 16; F = 2; log2_T = 19; N_min = 16
        per_level_scale = np.exp(np.log(2048*self.scale/N_min)/(L-1))

        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32, # 16 for dir encoding, 16 for geometric features
            n_output_dims=1+15, # 1 for sigmas, 15 for geometric features
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
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
                "n_neurons": 64,
                "n_hidden_layers": 2,
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
        #TODO edit this to include color

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
    pass


#TODO
class Trainer():
    pass



def main():
    model = INGP()
    print(model)
    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda')
    lr = 1e-3
    iters = 1e4
    workspace_dir = 'workspace'

    # Training
    optimizer = lambda model: torch.optim.Adam(model.get_params(lr), betas=(0.9, 0.99), eps=1e-15)

    train_loader = NerfDataset(device=device, type='train').dataloader()

    # decay to 0.1 * init_lr at last iter step
    scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / iters, 1))

    metrics = [PSNRMeter()]
    trainer = Trainer('ngp', model, device=device, workspace=workspace_dir, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=True, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=True, eval_interval=50)

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
    trainer = Trainer('ngp', model, device=device, workspace=workspace_dir, criterion=criterion, fp16=True, metrics=metrics, use_checkpoint=True)

    
    test_loader = NerfDataset(device=device, type='test').dataloader()

    if test_loader.has_gt: # check for ground truth
        trainer.evaluate(test_loader) # blender has gt, so evaluate it.

    trainer.test(test_loader, write_video=True) # test and save video
    
    trainer.save_mesh(resolution=256, threshold=10)