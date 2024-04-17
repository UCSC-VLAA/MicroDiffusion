import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import imageio

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.pose_utils import create_spiral_poses
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.lie_group_helper import convert3x4_4x4
from models.nerf_models import OfficialNerfMean
from models.intrinsics import LearnFocal
from models.poses import LearnPose
from utils.pos_enc import encode_position
import torch.nn.functional as F
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, action='store_true')
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=64, help='network hidden unit dimensions')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=20, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=False, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_load_sorted', type=bool, default=False)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--spiral_mag_percent', type=float, default=50, help='for np.percentile')
    parser.add_argument('--spiral_axis_scale', type=float, default=[1.0, 1.0, 1.0], nargs=3,
                        help='applied on top of percentile, useful in zoom in motion')
    parser.add_argument('--N_img_per_circle', type=int, default=60)
    parser.add_argument('--N_circle_traj', type=int, default=2)

    parser.add_argument('--ckpt_dir', type=str, default=r"./neurons_6_3")#'./logs/any_folder/any_folder_demo/desk/lr_0.001_gpu0_seed_17_resize_4_Nsam_128_Ntr_img_-1_freq_20__230927_0234')
    parser.add_argument('--steplength', type=int, default=6, help="step length")
    parser.add_argument('--path', type=str, default=-2, help="./neurons.npy")
    return parser.parse_args()

def get_all_gt_img(scene_train, z_total=130,z_sample=6):
    ids = np.arange(0,z_total,1)
    Total_gt = []
    for z in ids:
        img = scene_train.imgs[z].unsqueeze(0) # (H, W, 3)    
        Total_gt.append(img)
    Total_gt = torch.cat(Total_gt,dim=0)
    Total_gt=Total_gt
    return Total_gt

def get_all_pos_enc(H,W,z_total=130,z_sample=9):
    ids = np.arange(0,z_total)
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

        # encode position: (H, W, N_sample, (2L+1)*C = 63)
        # pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)
        Total_pos_enc.append(sample_pos)
    Total_pos_enc = torch.cat(Total_pos_enc,dim=0)
    Total_pos_enc=Total_pos_enc.cuda()
    # print(Total_pos_enc.shape)
    return Total_pos_enc    # exit()



def test_one_epoch(scene_train, near, far, model, my_devices, args):
    model.eval()

    z_sample = args.steplength
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    z_total = N_img
    L2_loss_epoch = []
    total_pos_enc = get_all_pos_enc(H,W,z_total,z_sample)
    ground = get_all_gt_img(scene_train,z_total,z_sample)
    print("total",ground.shape, N_img)
    # shuffle the training imgs
    test = z_sample
    L2_loss = 0
    result = []
    gt = []
    for i in range(int(z_total)//test+1):
        print(i*test, i*test + test)
        test = z_sample

        total_tmp = total_pos_enc[i*test:(i+1)*test,:,:,:].clone().cuda()
        test = min(test, ground.shape[0] - i*test)

        ground_tmp = ground[i*test:(i+1)*test,:,:,:].clone().cuda()
        
        print("ground_tmp", ground_tmp.shape)
        test = ground_tmp.shape[0]
        rendered = model(total_tmp,args, None,z_total,z_sample,test=True)
       # ren = torch.zeros_like(rendered[0,:,:,:,:])
       # for i in range(z_sample//2):
       #     print(i,0.3*(0.5**(z_sample//2 - i-1)))
       #     rendered += rendered[i,:,:,:,:] * 0.3*(0.5**(z_sample//2 - i))
# +  rendered[1,:,:,:,:] *0.3 + \
        #    rendered += rendered[-i,:,:,:,:] * 0.3*(0.5**(z_sample//2-i))
# + \
        rendered = rendered[1,:,:,:,:]*0.15 + rendered[0,:,:,:,:]*0.05 + rendered[2,:,:,:,:]*0.3 + \
            rendered[3,:,:,:,:]*0.3 + rendered[4,:,:,:,:]*0.15 + rendered[5,:,:,:,:]*0.05 
    
        print("rendered",rendered.shape)

        for k in range(test):
            print(k)
            r = rendered[k,:,:,0]
            r = (r - r.min())/(r.max()-r.min())
            r = r.cpu().numpy()
            result.append(r)
            g = ground_tmp[k,:,:,0].cpu().numpy()
            g = (g - g.min())/(g.max()-g.min())
            gt.append(g)        
            plt.subplot(121) 
            plt.xticks([]) 
            plt.yticks([]) 
            plt.axis('off') 
            plt.imshow(r)
            plt.subplot(122)
            plt.imshow(g)
            # plt.show()
            plt.xticks([]) 
            plt.yticks([]) 
            plt.axis('off') 
            # plt.savefig("t_"+str(i*test+k)+"dt.jpg",bbox_inches = "tight")
    gt = np.array(gt)
    result = np.array(result)
    
    print(gt.shape)
    print(result.shape)

    np.save("gt.npy",gt)
    np.save("inr.npy",result) 
    return L2_loss


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    test_dir = Path(os.path.join(args.ckpt_dir, 'render_spiral'))
    img_out_dir = Path(os.path.join(test_dir, 'img_out'))
    depth_out_dir = Path(os.path.join(test_dir, 'depth_out'))
    video_out_dir = Path(os.path.join(test_dir, 'video_out'))
    test_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    '''Load scene meta'''
    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted,
                                      path = args.path,
                                      load_img=False)

    print('H: {0:4d}, W: {1:4d}.'.format(scene_train.H, scene_train.W))
    print('near: {0:.1f}, far: {1:.1f}.'.format(scene_train.near, scene_train.far))

    '''Model Loading'''
    dir_enc_in_dims = 0
    
    model = OfficialNerfMean(512, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)
    model = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_nerf.pth'), model, map_location=my_devices)

   
    test_one_epoch(scene_train, scene_train.near, scene_train.far,
                            model, my_devices, args)


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)