import torch
import  torch.nn as nn
import  math
import  torch.nn.functional as F

import numpy as np


class PositionEncode(nn.Module):
    def __init__(self, d_model = 64):
        super(PositionEncode, self).__init__()
        self.d_model = d_model

    def forward(self, inp):
        B , _ , h, w = inp.size()
        device = inp.device
        x = torch.zeros((self.d_model,1,w),requires_grad=False)
        y = torch.zeros((self.d_model,h,1),requires_grad=False)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) *  ## 10000**（2i/d_model)
                             -(math.log(10000.0) / self.d_model))

        x_position = torch.arange(0,w).unsqueeze(0)  # 1 * w
        y_position = torch.arange(0,h).unsqueeze(1)  # h * 1

        x[0::2,:,:] = torch.sin(x_position.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        x[1::2,:,:] = torch.cos(x_position.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))

        y[0::2,:,:] = torch.sin(y_position.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        y[1::2,:,:] = torch.cos(y_position.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))

        x = x.repeat(1,h,1)
        y = y.repeat(1,1,w)
        pos = torch.cat((x,y),dim=0).unsqueeze(0).repeat(B,1,1,1).to(device
                                                                     )
        return pos


class ECRF(nn.Module):
    def __init__(self,compability_channel, kernel_size = 3, dilation = 2, stride = 1, channel = 32):
        super(ECRF, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.pos_embedding  = PositionEncode(d_model=channel)
        self.pos_conv = nn.Conv2d(kernel_size=1,
                                  in_channels=channel*2,out_channels=channel,bias=False)

        self.img_conv = nn.Conv2d(kernel_size=1,
                                  in_channels=3,out_channels=channel,bias=False)

        self.comp_conv = nn.Conv2d(kernel_size=1, in_channels=compability_channel,
                                   out_channels=1,bias=False)

    def feature_compatilibity(self, inp):
        unfold_fea = F.unfold(inp,kernel_size=self.kernel_size,dilation=self.dilation,
                 stride= self.stride,padding=2)

        B,C,H,W = inp.size()

        unfold_fea = unfold_fea.reshape(B,C,-1,H,W) ## B C K*K H W
        comp_score = []
        for i in range(self.kernel_size*self.kernel_size):
            comp_score.append(self.comp_conv(unfold_fea[:,:,i,:,:]))  # B K 1 H W

        comp_score = torch.stack(comp_score,dim=2) # B 1 K*K H W
        comp_score = torch.sigmoid(comp_score).squeeze(1)
        return comp_score, unfold_fea


    def forward(self, inp, img):

        B, C_f, H_f, W_f = inp.shape

        pos_embd = self.pos_embedding(inp)
        pos_embd = self.pos_conv(pos_embd)

        resize_img = F.interpolate(img, size=(H_f, W_f), mode="bilinear")
        resize_img = self.img_conv(resize_img)

        em_pos_inp = torch.cat([resize_img,pos_embd],dim=1)  # B Cf+C2 W H
        B, C , H , W = em_pos_inp.shape
        unfold_em_pos_inp = F.unfold(em_pos_inp,kernel_size=self.kernel_size,
                                     dilation=self.dilation,stride=self.stride, padding=2)

        unfold_em_pos_inp = unfold_em_pos_inp.reshape(B,C,-1,H,W)
        feature_score = (unfold_em_pos_inp * em_pos_inp.unsqueeze(2)).sum(dim=1) # B K*K H W
        # feature_score[:,self.kernel_size*self.kernel_size//2,:,:] = 0
        comp_score, unfold_fea = self.feature_compatilibity(inp)
        phi = (comp_score * feature_score).unsqueeze(1)  # B 1 K*K H W

        weighted_fea = (unfold_fea * phi).sum(dim=2)
        return weighted_fea


class SupAux(nn.Module):
    def __init__(self,weight=0.1):
        super(SupAux, self).__init__()
        self.weight = weight

    def forward(self, inp, superpixel):
        # superpixel
        max_ele = self.max_element(superpixel)

        B,C,H,W = inp.size()
        device = inp.device
        superpixel_tensor = torch.from_numpy(np.array(superpixel)).to(device)
        superpixel_tensor.requires_grad = False
        superpixel_tensor = F.interpolate(superpixel_tensor.unsqueeze(1),size=(H,W),mode='nearest') # B 1 H W

        for i in range(max_ele):
            mask_i = (superpixel_tensor == i) # B 1 H W
            sp_fea = mask_i * inp  # B C H W
            sp_fea_mean = (sp_fea.sum(dim=3,keepdim=True).sum(dim=2,keepdim=True)
                           ) / (
                    mask_i.sum(dim=3,keepdim=True).sum(dim=2,keepdim=True)
                    + 1e-5) # B C 1 1

            inp = inp + self.weight * sp_fea_mean * mask_i

        return inp

    def max_element(self,superpixel):
        max_ele = 0
        for sp in superpixel:
            m = np.max(sp)
            if max_ele < m:
                max_ele = m
        return max_ele

## Note that this message passing layer should be added before the final classification (linear) layer
## For more details, please see our paper

class MessagePass(nn.Module):
    def __init__(self,comp_channel,if_sp):
        super(MessagePass, self).__init__()
        self.ecrf = ECRF(compability_channel=comp_channel)
        self.if_sp = if_sp  # if use superpixel in the process
        if if_sp:
            self.sp_aux = SupAux(weight=0.1)
        else:
            self.sp_aux = None

    def forward(self, inp, img, superpixel):
        weight_fea = self.ecrf(inp,img)
        if self.if_sp:
            inp = self.sp_aux(inp, superpixel)
        inp = inp + 0.2 * weight_fea
        return inp


# if __name__ == "__main__":
#     mp = MessagePass(comp_channel=256,if_sp=True).cuda()
#     x = torch.rand([5,256,64,64]).cuda()
#     img = torch.rand(5,3,256,256).cuda()
#     import os
#     superpixel_path = "./ADEChallengeData2016/superpixel/training"
#     file = os.listdir(superpixel_path)[:5]
#     assert len(file) == 5
#     import skimage.io as io
#     sp = []
#     for i in file:
#         sp_img = io.imread(os.path.join(superpixel_path, i))
#         sp_img = np.resize(sp_img,(256,256))
#         sp.append(sp_img)
#     sp = np.array(sp)
#     x_new = mp(x,img,sp)
#     a = x_new.mean()
#     a.backward()



