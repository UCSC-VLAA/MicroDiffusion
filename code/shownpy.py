import numpy as np 
import torch
import torch.nn.functional as F
from skimage import io
def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB
from PIL import Image 
im = io.imread("YFP_data.tif")
gt = im.astype(np.float32).transpose(1,2,0)
print(gt.shape)
np.save("YFP_data.npy",gt)
# gt = gt.transpose(2,0,1)
# gt = torch.tensor(gt).unsqueeze(-1).cuda()
# gt = resize_imgs(gt,128,128)
# gt = gt.squeeze().cpu().numpy()
# print("gt",gt.shape)
# gt = (gt - np.min(gt))/(np.max(gt)-np.min(gt))



# #INR
# imgs = np.load(r"E:/test_result_3.npy")
# print("imgs",imgs.shape)
# imgs = imgs[1:]
# imgs = (imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))
# print(np.max(gt),np.min(gt))


# #inter
# bases = []
# for i in range(0,130,5):
#     bases.append(np.mean(gt[i:i+5,:,:],axis=0,keepdims=True))
# print(bases[0].shape)

# result = []
# for i in range(0,125//5):
#     print(i)
#     result.append(1*bases[i]+0*bases[i+1])
#     result.append(0.75*bases[i]+0.25*bases[i+1])
#     result.append(0.5*bases[i]+0.5*bases[i+1])
#     result.append(0.25*bases[i]+0.75*bases[i+1])
#     result.append(0*bases[i]+1*bases[i+1])

# inter = np.concatenate(result,axis=0)
# inter = (inter - np.min(inter))/(np.max(inter)-np.min(inter))

# print(inter.shape)




# import imageio
# import numpy as np
# img_paths = []
# for i in range(3,128):
#     img_paths.append(r"E:\UCSC_related\sample\generated_2001_"+str(i)+"_pict.png")
# gif_images = []
# for path in img_paths:
#     im = imageio.imread(path)
#     print(type(im),im.dtype)
#     print(im.shape)
#     gif_images.append(im)
# gif_images = np.array(gif_images)
# print(gif_images.shape)

# diff = np.mean(gif_images, axis=-1)
# diff = (diff - np.min(diff))/(np.max(diff)-np.min(diff))

# print("diff",diff.shape)



# # gt = gt[2:gt+128]

# def psnr(img1,img2):
#     mse = np.mean((img1-img2)**2)
#     return 20 * np.log10(255.0/np.sqrt(mse))

# def ssim(y_true , y_pred):
#     u_true = np.mean(y_true)
#     u_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     std_true = np.sqrt(var_true)
#     std_pred = np.sqrt(var_pred)
#     c1 = np.square(0.01*7)
#     c2 = np.square(0.03*7)
#     ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim / denom

# gt = (gt*255).astype(np.uint8)
# diff = (diff*255).astype(np.uint8)
# imgs = (imgs*255).astype(np.uint8)
# inter = (inter*255).astype(np.uint8)

# for i in range(0,6):
#     gt_tmp = gt[i:i+125]
#     print(gt_tmp.shape,i+125)
#     print(psnr(diff,gt_tmp), ssim(gt_tmp,diff))
#     print(psnr(imgs,gt_tmp), ssim(gt_tmp,imgs))
#     print(psnr(inter,gt_tmp), ssim(inter,imgs))


    # print(np.sqrt(np.sum((gt_tmp-diff)**2)/125))
    # print(np.sqrt(np.sum((gt_tmp-imgs)**2)/125))

# import matplotlib.pyplot as plt
# for i in range(diff.shape[0]):
#     plt.subplot(121)
#     plt.imshow(diff[i ,:,:])
#     plt.subplot(122)
#     plt.imshow(gt[i+3 ,:,:])
#     plt.title(label=str(i))
#     plt.show()
#     plt.clf()