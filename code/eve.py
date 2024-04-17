import matplotlib.image as mpimg
import numpy as np
import torch
import os
import cv2
import re
from skimage.metrics import structural_similarity as ssim

# 读取图片

inr =  torch.tensor(np.load("path_to_inr_result")).unsqueeze(0)
gt = torch.tensor(np.load("path_to_gt_result")).unsqueeze(0)


def extract_number(filename):
    s = re.search(r'\d+', filename)
    return int(s.group()) if s else None

def dice_coeff(pred, target):
    
    smooth = 1e-6
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)  # Flatten
    m2 = target.contiguous().view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def psnr(img1,img2):

    mse = np.mean((img1-img2)**2)

    return 20 * np.log10(255.0/np.sqrt(mse))

def ca_psnr(array1,array2):

    total = []
    for i in range(array1.shape[0]):
        total.append(psnr(array1[i],array2[i]))
    return np.array(total).mean()



def ca_ssim(array1,array2):

    total_Luminance = []
    total_Contrast = []
    total_Structure = []
    total = []
    for i in range(array1.shape[0]):
        m = ssim(array1[i],array2[i])

        total.append(m)
    return np.array(total).mean()

image_folder_path = "path to generate result"
image_filenames = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

image_filenames = sorted(image_filenames, key=extract_number)

image_list_bi = []
image_list = []
for filename in image_filenames:
   
    full_path = os.path.join(image_folder_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_list.append(image)
    image_list_bi.append(thresh)

inr_list = []
gt_list = []
for i in range(inr.shape[1]):

    _, thresh = cv2.threshold(((inr[0,i]*255).numpy()).astype(np.uint16), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inr_list.append(thresh)
    _, thresh = cv2.threshold(((gt[0,i]*255).numpy()).astype(np.uint16), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gt_list.append(thresh)


image_array = np.array(image_list)
gt_array = np.array(gt[0])*255
inr_array = np.array(inr[0])*255

image_tensor = (torch.tensor(np.array(image_list_bi).astype(np.int16))/255).unsqueeze(0)
gt_tensor = (torch.tensor(np.array(gt_list).astype(np.int16))/255).unsqueeze(0)
inr_tensor = (torch.tensor(np.array(inr_list).astype(np.int16))/255).unsqueeze(0)


print("image")
print(ca_ssim(gt_array,image_array))
print(ca_psnr(image_array,gt_array))
print(dice_coeff(image_tensor,gt_tensor).item())

print("INR")
print(ca_ssim(gt_array,inr_array))
print(ca_psnr(inr_array,gt_array))
print(dice_coeff(inr_tensor,gt_tensor).item())



