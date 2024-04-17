import torch
import numpy as np

import numpy as np

import torch
import torch.nn as nn



############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self):
        self.B = torch.randn((256, 3)) * 4
        self.B = self.B.cuda()

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding

enc = Positional_Encoder()
enc = torch.load("./enc.pt")

def encode_position(input, levels=None, inc_input=None):
    """
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.

    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """
    # print("input",input.shape,input[:, : , :, :2].shape)
    # # this is already doing 'log_sampling' in the official code.
#    enc = Positional_Encoder()
 #   enc = torch.load("./enc.pt")
    global enc 
    result_list = enc.embedding(input)
    # result_list = [input] if inc_input else []
    # for i in range(levels):
        # temp = 2.0**i * input  # (..., C)
        # result_list.append(torch.sin(temp))  # (..., C)
        # result_list.append(torch.cos(temp))  # (..., C)
# 
    # result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    
    # print("input",input.shape)
    # if len(input.shape) == 4:
    #     s = np.sin(np.arange(0, 180, 45) * np.pi / 180)[:, np.newaxis]
    #     c = np.cos(np.arange(0, 180, 45) * np.pi / 180)[:, np.newaxis]
    #     fourier_mapping = np.concatenate((s, c), axis=1).T
    #     fourier_mapping = torch.tensor(fourier_mapping).float().to(input.device)
        
    #     xy_freq = torch.matmul(input[:, : , :, :2], fourier_mapping)

    #     for l in range(8):
    #         cur_freq = torch.concat(
    #             [
    #                 torch.sin(2 ** l * np.pi * xy_freq),
    #                 torch.cos(2 ** l * np.pi * xy_freq),
    #             ],
    #             axis=-1,
    #         )
    #         if l == 0:
    #             tot_freq = cur_freq
    #         else:
    #             tot_freq = torch.concat([tot_freq, cur_freq], axis=-1)
        
    #     # print(tot_freq.shape)

    #     for l in range(6):
    #         cur_freq = torch.concat(
    #             [
    #                 torch.sin(2 ** l * np.pi * input[:, : , :, 2][:, :,:,None]),
    #                 torch.cos(2 ** l * np.pi * input[:, : , :, 2][:, :,:,None]),
    #             ],
    #             axis=-1,
    #         )
    #     # print(cur_freq.shape)
    #         tot_freq = torch.concat([tot_freq, cur_freq], axis=-1)
    #     print("tot_freq",tot_freq.shape)
    #     return tot_freq  # (..., C*(2L+1))
    # else:
    #     s = np.sin(np.arange(0, 180, 45) * np.pi / 180)[:, np.newaxis]
    #     c = np.cos(np.arange(0, 180, 45) * np.pi / 180)[:, np.newaxis]
    #     fourier_mapping = np.concatenate((s, c), axis=1).T
    #     fourier_mapping = torch.tensor(fourier_mapping).float().to(input.device)
        
    #     xy_freq = torch.matmul(input[: , :, :2], fourier_mapping)

    #     for l in range(8):
    #         cur_freq = torch.concat(
    #             [
    #                 torch.sin(2 ** l * np.pi * xy_freq),
    #                 torch.cos(2 ** l * np.pi * xy_freq),
    #             ],
    #             axis=-1,
    #         )
    #         if l == 0:
    #             tot_freq = cur_freq
    #         else:
    #             tot_freq = torch.concat([tot_freq, cur_freq], axis=-1)
        
    #     # print(tot_freq.shape)

    #     for l in range(6):
    #         cur_freq = torch.concat(
    #             [
    #                 torch.sin(2 ** l * np.pi * input[:, : , 2][:,:,None]),
    #                 torch.cos(2 ** l * np.pi * input[:, :, 2][:, :,None]),
    #             ],
    #             axis=-1,
    #         )
    #     # print(cur_freq.shape)
    #         tot_freq = torch.concat([tot_freq, cur_freq], axis=-1)
    #     print("tot_freq",tot_freq.shape)
    return result_list  # (..., C*(2L+1))i


#if __name__=="__main__":
#torch.save(enc,"~/enc.pt")
