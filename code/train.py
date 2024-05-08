import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utilsdiff import get_named_beta_schedule
from embedding import ConditionalEmbedding, INREmbedding
from Scheduler import GradualWarmupScheduler
# from dataloader_cifar import load_data, transback
from dataloader import DataLoaderAnyFolder
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from pos_enc import encode_position
from utils.training_utils import set_randomness, load_ckpt_to_net
from models.nerf_models import OfficialNerfMean

# 128*128*130
def get_all_pos_enc(H,W,z_total=130,z_sample=9):
    ids = np.arange(0,z_total)
    Test_X = torch.arange(-1,1,2.0/H)
    Test_Y = torch.arange(-1,1,2.0/W)
    Total_pos_enc = []
    for z in ids:
        # print(z)
        Test_Z = z * 2.0/z_total -1.0
        sample_pos = torch.zeros(1,H,W,3)
        for i in range(H):
            for j in range(W):
                sample_pos[0,i,j,0]= Test_X[i]
                sample_pos[0,i,j,1]= Test_Y[j]
                sample_pos[0,i,j,2]= Test_Z

        sample_pos = sample_pos.cuda()
        Total_pos_enc.append(sample_pos)
    Total_pos_enc = torch.cat(Total_pos_enc,dim=0)
    Total_pos_enc=Total_pos_enc.cuda()
    return Total_pos_enc 

#total_pos_enc = get_all_pos_enc(128,128,130,6)
def train(params:argparse.Namespace):
    # initialize settings
    # init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = 0
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    data_loader = DataLoaderAnyFolder(params.gt_path)
    N_img, H, W = data_loader.N_imgs, data_loader.H, data_loader.W
    
    #Diffusion Net
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim + 128,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    
    #POS MLP; 21->32
    cemblayer = ConditionalEmbedding(21, params.cdim, params.cdim).to(device)
    
    #Image Encoder RN18; 1*128*128 => 128
    inremblayer = INREmbedding(feature_emb=128).to(device)
    
    # INR Pretrained
    dir_enc_in_dims = 0
    model = OfficialNerfMean(512, dir_enc_in_dims, params.hidden_dims)
    model = model.to(device=torch.device("cuda"))
    model = load_ckpt_to_net(os.path.join(params.ckpt_dir, params.dataset_name+'.pth'), model, map_location=torch.device("cuda"))
    
    
    # load last epoch
    lastpath = os.path.join(params.moddir,'ckpt_2001_checkpoint.pt')
    if os.path.exists(lastpath):
        print("EXIST")
        lastepc = 2001   
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
        inremblayer.load_state_dict(checkpoint['inremb'])
        model.load_state_dict(checkpoint['inr']),
    else:
        lastepc = 0
        
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)

    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
 
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters(),
                        inremblayer.parameters(),
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            warm_epoch = params.epoch ,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    #FIXME Hardcoded from INR model 
    z_total = 130 
    z_sample = 6

    total_pos_enc = get_all_pos_enc(H,W,z_total,z_sample)

    #for epc in range(0):
    for epc in tqdm(range(lastepc, params.epoch)):
        diffusion.model.train()
        cemblayer.train()
        model.train()
        inremblayer.train()

        for z_space in range(3, data_loader.N_imgs-3,6):
            z = 2*z_space / data_loader.N_imgs - 1.0 #Normalize -1, 1

            total_tmp = total_pos_enc[z_space-3:z_space+3,:,:,:].clone().cuda()
            rendered = model(total_tmp, params, None,z_total,z_sample,test=True)
            rendered = rendered[0,:,:,:,:] *0.05 +  rendered[1,:,:,:,:] *0.15 + \
                rendered[2,:,:,:,:]*0.3 + \
                rendered[3,:,:,:,:]*0.3 + \
                rendered[4,:,:,:,:]*0.15+ \
                rendered[5,:,:,:,:]*0.05
            # Condition from INR

            # Artificat Input image; Ground truth
            img = data_loader.imgs[z_space-3:z_space+3].mean(dim=0).squeeze(0)
            img = img.permute(2,0,1)
            img = img.unsqueeze(0)

            b = img.shape[0]
            optimizer.zero_grad()
            x_0 = img.to(device)
            cembs = []
            inters = []
            inr_feature = inremblayer(x_0)
            for i in range(-3,3):
                z_tmp = z + i* (2.0/data_loader.N_imgs)
                lab = encode_position(torch.tensor([[z_tmp]]).float()) 
                lab = lab.to(device)
                cemb = cemblayer(lab)
                inter = rendered[i+3].permute(2,0,1)
                cembcat = torch.cat([cemb,inr_feature],1)
                cembs.append(cembcat)
                inters.append(inter)
            loss, generated = diffusion.trainloss(inters,x_0, cembs = cembs)
            print(loss.item())
            loss.backward()
            optimizer.step()
            
            print("Total:{:4f}".format(loss.item()))

        warmUpScheduler.step()
    if True:
        checkpoint = {
                            'net':diffusion.model.state_dict(),
                            'cemblayer':cemblayer.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'inr':model.state_dict(),
                            'inremb':inremblayer.state_dict(),
                            'scheduler':warmUpScheduler.state_dict()
                        }
        torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
        torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        #torch.save(torch.cat(all_samples,dim=0),"Result.pth") 
        torch.cuda.empty_cache()
        
    if True:
        diffusion.model.eval()
        cemblayer.eval()
        model.eval()
        inremblayer.eval()
        # generating samples
        # The model generate 80 pictures(8 per row) each time
        # pictures of same row belong to the same class
        all_samples = []
        for z_space in range(3,data_loader.N_imgs-3,1):# data_loader.N_imgs):
            if z_space+4>data_loader.N_imgs:
                break
            with torch.no_grad():
                z = 2*z_space / data_loader.N_imgs - 1.0

                #INR
                total_tmp = total_pos_enc[z_space-3:z_space+3,:,:,:].clone().cuda()
                rendered = model(total_tmp, params, None,z_total,z_sample,test=True)
                rendered = rendered[0,:,:,:,:] *0.05 +  rendered[1,:,:,:,:] *0.15 + \
                    rendered[2,:,:,:,:]*0.3 + \
                    rendered[3,:,:,:,:]*0.3 + \
                    rendered[4,:,:,:,:]*0.15+ \
                    rendered[5,:,:,:,:]*0.05
                    
                img = data_loader.imgs[z_space-3:z_space+3].mean(dim=0).squeeze(0)
                img = img.permute(2,0,1)
                img = img.unsqueeze(0)
                x_0 = img.to(device)                     
                inters = rendered[3].permute(2,0,1).unsqueeze(dim=0)        
                inr_feature = inremblayer(x_0).squeeze()
                
                lab = encode_position(torch.tensor([[z]]).float())
                lab = lab.to(device)
                cemb = cemblayer(lab).squeeze()
                cembcat = torch.cat([cemb,inr_feature]).unsqueeze(dim=0)
                genshape = (1 , 1 , 128, 128)
                if params.ddim:
                    generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cembcat)
                else:
                    generated = diffusion.sample(genshape,inters ,cemb = cembcat)

                samples = generated
                all_samples.append(samples)
                save_image(samples, os.path.join(params.samdir, f'{z_space-3}.png'), nrow = params.genbatch // params.clsnum)


def main():

    import os

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'

    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')
    parser.add_argument('--dataset_name',type=str,default="Dendrite",help='dataset_name')
    parser.add_argument('--gt_path',type=str,default="Dendrite",help='gt_path')
    parser.add_argument('--ckpt_dir', type=str, default=r'inr_ckpt_path')
    parser.add_argument('--moddir',type=str,default='work_dir',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--epoch',type=int,default=2001,help='epochs for training') 
    parser.add_argument('--batchsize',type=int,default=1,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=1, help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=1,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=32,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=1,help='epoch interval between two evaluations')
    parser.add_argument('--genbatch',type=int,default=1,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=1,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--hidden_dims', type=int, default=64, help='network hidden unit dimensions')
    

    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
