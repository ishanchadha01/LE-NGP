import torch
import os
from tqdm import tqdm

from metrics import PSNRMeter


#TODO
class NerfDataset():
    pass


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


