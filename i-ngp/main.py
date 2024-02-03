import torch

from models import INGP
from metrics import PSNRMeter
from data import NerfDataset, Trainer


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