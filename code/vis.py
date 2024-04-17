from torchvision.utils import save_image
import torch
import os

inr =  torch.tensor(np.load("./beads_inr.npy")).unsqueeze(0)
gt = torch.tensor(np.load("./beads_gt.npy")).unsqueeze(0)

for i in range(inr.shape[1]):

    save_image(gt[0,i], os.path.join("../data", f'{i}.png'))   
    save_image(inr[0,i], os.path.join("../data", f'{i}.png'))