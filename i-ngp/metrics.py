import torch


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
        
