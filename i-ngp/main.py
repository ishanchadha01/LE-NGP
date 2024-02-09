import torch
import numpy as np
import torch.distributed as distributed

from models import INGP
from metrics import PSNRMeter
from data import NerfDataset, Trainer


def main(root_path):
    distributed.init_process_group(backend='nccl', rank=0, world_size=1)  # Use 'nccl' as the backend if you're using GPUs, rank is local rank so 0, world size is num GPUs
    model = INGP()
    print(model)
    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda')
    lr = 1e-3
    iters = 1e4

    # Training
    optimizer = lambda model: torch.optim.Adam(model.module.get_params(lr), betas=(0.9, 0.99), eps=1e-15)
    train_loader = NerfDataset(device=device, type='train', path=root_path).dataloader()
    # train_loader = DataLoader(device=device, type='train', path='root_path') # TODO: make dataloader in a better way, current way is kinda dumb

    # decay to 0.1 * init_lr at last iter step
    scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / iters, 1))
    metrics = [PSNRMeter()]
    trainer = Trainer(model, 
                      lr_scheduler=scheduler, 
                      criterion=criterion, 
                      optimizer=optimizer,
                      device=device, 
                      metrics=metrics)
    valid_loader = NerfDataset(device=device, type='val', downscale=1, path=root_path).dataloader()

    max_epoch = np.ceil(iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, max_epoch)

    # also test
    test_loader = NerfDataset(device=device, type='test', path=root_path).dataloader()
    
    if test_loader.has_gt:
        trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
    trainer.test(test_loader, write_video=True) # test and save video
    
    # trainer.save_mesh(resolution=256, threshold=10) TODO: implement

    # Testing
    # Can make a new trainer here if params are different
    
    test_loader = NerfDataset(device=device, type='test').dataloader()

    if test_loader.has_gt: # check for ground truth
        trainer.evaluate(test_loader) # blender has gt, so evaluate it.

    trainer.test(test_loader, write_video=True) # test and save video
    
    # trainer.save_mesh(resolution=256, threshold=10) TODO: implement


if __name__=='__main__':
    #TODO add argparser
    root_path = "/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/le-ngp/data/cecum_t1_a"
    main(root_path)