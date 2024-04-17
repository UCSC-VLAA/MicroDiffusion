import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint,load_ckpt_to_net
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling_ndc, volume_rendering
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from models.nerf_models import OfficialNerf,OfficialNerfMean
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--eval_interval', default=100, type=int, help='run eval every this epoch number')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='inr')

    parser.add_argument('--nerf_lr', default=0.001, type=float)
    parser.add_argument('--nerf_milestones', default=list(range(0, 10000, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.9954, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=True, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_lr', default=0.001, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=64, help='network hidden unit dimensions')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=20, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_load_sorted', type=bool, default=True)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')

    parser.add_argument('--alias', type=str, default='', help="experiments alias")

    parser.add_argument('--steplength', type=int, default=6, help="step length")
    parser.add_argument('--overlap', type=int, default=-2, help="overlapping")
    parser.add_argument('--path', type=str, default=-2, help="./neurons.npy")

    return parser.parse_args()


def gen_detail_name(args):
    outstr = 'lr_' + str(args.nerf_lr) + \
             '_gpu' + str(args.gpu_id) + \
             '_seed_' + str(args.rand_seed) + \
             '_resize_' + str(args.resize_ratio) + \
             '_Nsam_' + str(args.num_sample) + \
             '_Ntr_img_'+ str(args.train_img_num) + \
             '_freq_' + str(args.pos_enc_levels) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr


# [  5  14  23  32  41  50  59  68  77  86  95 104 113 122]
def get_one_pos_enc(H,W,z=0,z_total=130):
    Test_X = torch.arange(-1,1,2.0/H)
    Test_Y = torch.arange(-1,1,2.0/W)
    Test_Z = z * 2.0/z_total -1.0
    sample_pos = torch.zeros(1,H,W,3)
    for i in range(H):
        for j in range(W):
            sample_pos[0,i,j,0]= Test_X[i]
            sample_pos[0,i,j,1]= Test_Y[j]
            sample_pos[0,i,j,2]= Test_Z

    sample_pos = sample_pos.cuda()

    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)
    #print(pos_enc.shape)
    return pos_enc

def get_all_pos_enc(H,W,z_total=130,z_sample=5, overlap = True , overlapping = None):
    if overlap:
        if overlapping == None:
            overlapping = z_sample//2

        step = z_sample - overlapping
        ids = np.arange(z_sample//2,z_total-z_sample//2,step)
    else:
        ids = np.arange(z_sample//2,z_total-z_sample//2,z_sample)

    Test_X = torch.arange(-1,1,2.0/H)
    Test_Y = torch.arange(-1,1,2.0/W)
    Total_pos_enc = []
    for z in ids:
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
    # print(Total_pos_enc.shape)
    return Total_pos_enc    # exit()

def get_all_gt_img(scene_train, z_total=130,z_sample=6, overlap = True, overlapping = None):
    if overlap:
        if overlapping == None:
            overlapping = z_sample//2

        step = z_sample - overlapping

        ids = np.arange(z_sample//2,z_total-z_sample//2,step)
    else:
        ids = np.arange(z_sample//2,z_total-z_sample//2,z_sample)

    Total_gt = []
    for z in ids:
        img = scene_train.imgs[z-z_sample//2:z+z_sample//2].mean(dim=0).unsqueeze(0) # (H, W, 3)    
        Total_gt.append(img)
    Total_gt = torch.cat(Total_gt,dim=0)
    Total_gt=Total_gt.cuda()
    return Total_gt




def train_one_epoch(z_sample, z_total, total_pos_enc, ground, scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
                    my_devices, args, rgb_act_fn):
    model.train()
    L2_loss_epoch = []

    #Bathwise here
    #print(total_pos_enc.shape)
    rendered, rgb, pos_enc_tmp = model(total_pos_enc, args, None,z_total,z_sample)
    pos_enc_tmp = pos_enc_tmp.transpose(0,1)
    L_rec = F.mse_loss(rendered, ground)
    L2_loss = L_rec

    L2_loss.backward()
    optimizer_nerf.step()
    optimizer_nerf.zero_grad()
    L2_loss_epoch.append(L2_loss.item())
    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/nerf_models.py', experiment_dir)
    shutil.copy('./tasks/any_folder/train.py', experiment_dir)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=str(experiment_dir))

    '''Data Loading'''
    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted,
                                      path = args.path)

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))

    # We have no eval pose in this any_folder task. Eval with a 4x4 identity pose.
    eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

    '''Model Loading'''
    dir_enc_in_dims = 0

    model = OfficialNerfMean(512, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)
    
    

    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)

    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones,
                                                          gamma=args.nerf_lr_gamma)

    '''Training'''
    z_sample = args.steplength
    overlapping = args.overlap
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    z_total = N_img
    total_pos_enc = get_all_pos_enc(H,W,z_total,z_sample,True,overlapping)
    ground = get_all_gt_img(scene_train,z_total,z_sample,True,overlapping)

    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        rgb_act_fn = torch.sigmoid
        train_epoch_losses = train_one_epoch(z_sample, z_total, total_pos_enc, ground, scene_train, optimizer_nerf, None, None,
                                             model, None, None, my_devices, args, rgb_act_fn)
        train_L2_loss = train_epoch_losses['L2']
        scheduler_nerf.step()

        train_psnr = mse2psnr(train_L2_loss)
        writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        writer.add_scalar('train/psnr', train_psnr, epoch_i)
        writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)
        logger.info('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))
        tqdm.write('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))

        if True:
            with torch.no_grad():
                # save the latest model
                save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                print("save",experiment_dir) 
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)
