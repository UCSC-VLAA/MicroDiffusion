import os

import torch
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import torch.nn.functional as F

def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB

def load_imgs(data_path):

    from os.path import dirname, join as pjoin
    import matplotlib.pyplot as plt
    import scipy.io as sio

    path = data_path
    img_list = np.load(path)


    H,W,N_imgs = img_list.shape
    img_list = img_list.reshape(H,W,N_imgs,1)
    img_list = img_list.transpose((2,0,1,3))
    img_list = img_list.astype(float)
    maxn = np.max(img_list)
    minn = np.min(img_list)
    img_list = (img_list - minn)/(maxn-minn)
    img_list = torch.from_numpy(img_list).float()  # (N, H, W, 1) torch.float32

    results = {
        'imgs': img_list,  # (N, H, W, 1) torch.float32
        'img_names': [],  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }

    return results


class DataLoaderAnyFolder:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self,data_path):
        """
        :param base_dir:
        :param scene_name:
        :param res_ratio:       int [1, 2, 4] etc to resize images to a lower resolution.
        :param start/end/skip:  control frame loading in temporal domain.
        :param load_sorted:     True/False.
        :param load_img:        True/False. If set to false: only count number of images, get H and W,
                                but do not load imgs. Useful when vis poses or debug etc.
        """
       
        self.load_img = True

        image_data = load_imgs(data_path)
        self.imgs = image_data['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = image_data['img_names']  # (N, )
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']
        print(self.ori_H,self.ori_W)

        self.res_ratio = 4

        # always use ndc
        self.near = 0.0
        self.far = 1.0
        self.H =128
        self.W = 128

        if self.load_img:
            self.imgs = resize_imgs(self.imgs, self.H, self.W)  # (N, H, W, 3) torch.float32


if __name__ == '__main__':
    base_dir = '/your/data/path'
    scene_name = 'LLFF/fern/images'
    resize_ratio = 8
    num_img_to_load = -1
    start = 0
    end = -1
    skip = 1
    load_sorted = True
    load_img = True

    scene = DataLoaderAnyFolder()
