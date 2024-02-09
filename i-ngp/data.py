import torch
from torch_ema import ExponentialMovingAverage
import os
from tqdm import tqdm
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import torch.distributed as distributed
import imageio
from torch.utils.data.distributed import DistributedSampler

from metrics import PSNRMeter
from utils import nerf_matrix_to_ngp, rand_poses, get_rays, linear_to_srgb, srgb_to_linear



class NerfDataset():
    def __init__(self, 
                 device="cuda", 
                 downscale=1,
                 path="./data",
                 preload=True, # preload data into GPU
                 camera_scale=1, # camera radius scale to make sure camera are inside the bounding box
                 offset=[0,0,0], # camera offset
                 bbox_scale=1, # bounding box half length, also used as the radius to random sample poses
                 fp16=True, # if preload, load into fp16
                 num_rays=4096,
                 rand_pose=-1,
                 error_map=True,
                 color_space="srgb",
                 patch_size=1, # [experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable
                 type='train', # train, val, test
                 n_test=10,
                 local_rank=0,
                 world_size=1
                 ):
        # Use transform files to initialize dataset by matching poses to image names

        self.device = device
        self.type = type 
        self.downscale = downscale
        self.root_path = path
        self.preload = preload 
        self.camera_scale = camera_scale
        self.offset = offset 
        self.bbox_scale = bbox_scale # not used ??
        self.fp16 = fp16
        self.error_map = error_map
        self.color_space = color_space
        self.patch_size = patch_size
        self.local_rank = local_rank
        self.world_size = world_size

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = num_rays if self.training else -1
        self.rand_pose = rand_pose

        # auto-detect transforms.json and split mode
        if os.path.exists(os.path.join(self.root_path, 'transforms_train.json')): #TODO: make a combined transforms json or separate transforms_train and transforms_test
            self.mode = 'colmap' # manually split, use view-interpolation for test
            with open(os.path.join(self.root_path, 'transforms_train.json'), 'r') as f: # load nerf-compatible format data
                transform = json.load(f)
        # dont use blender for now, TODO what is purpose of transforms_train?
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')
        
        # assume height and width are given in file
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        
        # read images
        frames = transform["frames"]
        
        if self.mode == 'colmap' and type == 'test':
            # for colmap, manually interpolate a test set
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.camera_scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.camera_scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]])) # rotation between poses
            slerp = Slerp([0, 1], rots) # spherical interpolation between poses

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.camera_scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)

        
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        # initialize error_map
        if self.training and self.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # TODO: can visualize poses here to debug

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def __len__(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose 
        return size


    def collate(self, index):
        B = len(index) # a list of length 1
        if self.rand_pose == 0 or index[0] >= len(self.poses): # random pose without ground truth images
            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for speedup
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)
            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.patch_size)
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }
        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']   
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, num_workers=0, sampler=DistributedSampler(self))
        # loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0) # nondistributed
        loader._data = self # need to access error_map & poses in trainer
        loader.has_gt = self.images is not None
        return loader


class Trainer():
    def __init__(self, 
                 model,
                 lr_scheduler, 
                 criterion,
                 optimizer,
                 device,
                 color_space="linear",
                 experiment_name="ngp",
                 workspace="./workspace", 
                 ema_decay=0.95, # for smoothing
                 fp16=True, 
                 scheduler_update_every_step=True, 
                 metrics=[PSNRMeter()], 
                 use_checkpoint="latest", 
                 eval_interval=50,
                 local_rank=0, # device id if doing distributed training
                 world_size=1, # number of processes to use for distributed trainer, total num of gpus
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
        self.local_rank = local_rank
        self.world_size = world_size
        self.color_space = color_space

        model.to(self.device) 
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) # TODO: maybe shouldnt do distributed training?

        if isinstance(criterion, nn.Module):
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
        self.best_path = os.path.join(self.ckpt_path, f"{experiment_name}.pth")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.load_ckpt(self.use_checkpoint) # can configure this to be latest, best, etc

        # can use a clip based loss? why would an image to text classifier provide good loss function for this?
        # TODO: maybe similar to text guided object detection like dreamfields

    def __del__(self):
        self.log_ptr.close()

    def log(self, *args):
        if self.local_rank==0 and self.log_ptr: # only log on local device if log_ptr exists
            print(*args, file=self.log_ptr)
            self.log_ptr.flush() # write to file immediately


    def train_one_epoch(self, train_loader):
        self.log(f"Training Epoch {self.epoch}")
        self.model.train() # Put model in training mode

        # Clear metrics
        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()
        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        # train_loader.sampler.set_epoch(self.epoch) #TODO: might not need to do this due to small world size
        pbar = tqdm(
            total=len(train_loader) * train_loader.batch_size, 
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        for data in train_loader:
            # update grid every 16 steps
            if self.model.module.cuda_ray_marching and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.module.update_extra_state() # update state info for nerf apart from just member vars of class, so this includes hash grids, resolutions, etc

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(preds, truths)
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(train_loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        if self.local_rank == 0:
            pbar.close()
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
            
    def eval_one_epoch(self, eval_loader):
        self.log(f"Eval Epoch {self.epoch}")
        name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm(total=len(eval_loader) * eval_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in eval_loader:
                self.local_step += 1
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*), basically mapreduce
                if self.world_size > 1:
                    distributed.all_reduce(loss, op=distributed.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    distributed.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    distributed.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    distributed.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                loss_val = loss.item()
                total_loss += loss_val

                if self.local_rank == 0:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.color_space == 'linear':
                        preds = linear_to_srgb(preds)
                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(eval_loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result)
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()
        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def train_step(self, data):
        # trace rays through image to render them
        rays_o = data["rays_o"]
        rays_d = data["rays_d"]
        images = data['images'] # [B, N, 3/4]
        B, N, C = images.shape #batches, imgs, channels

        if self.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.module.bg_radius > 0:
            bg_color = 1
        else: # train with random background color if not using a bg model and has alpha channel
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.module.render(rays_o, rays_d)
        pred_rgb = outputs['image']
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N], mse loss

        # patch-based rendering
        if self.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.patch_size, self.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.patch_size, self.patch_size, 3).permute(0, 3, 1, 2).contiguous()

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]
            # TODO: could save error map here to visualize and debug
            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()
        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape
        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        outputs = self.model.module.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = self.criterion(pred_rgb, gt_rgb).mean()
        return pred_rgb, pred_depth, gt_rgb, loss

    def test_step(self, data, bg_color=None):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']
        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        outputs = self.model.module.render(rays_o, rays_d, staged=True, bg_color=bg_color, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)
        return pred_rgb, pred_depth

    def train(self, train_loader, valid_loader, max_epochs):
        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.module.cuda_ray_marching:
            self.model.module.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader)
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

    def eval(self, loader, name=None):
        self.eval_one_epoch(loader, name)

    def test(self, test_loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Start Test, save results to {save_path}")
        pbar = tqdm.tqdm(total=len(test_loader) * test_loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        if write_video:
            all_preds = []
            all_preds_depth = []
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)
                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                pbar.update(test_loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    

    def save_ckpt(self):
        pass

    def load_ckpt(self, use_cpkt):
        pass

    def save_mesh(self):
        pass


