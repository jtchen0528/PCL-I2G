#%%
import torch
from torch import nn
from torch.nn import functional as F
import math

class NLBlockND(nn.Module):
    def __init__(self, in_channels=256):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()


        self.in_channels = in_channels
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions

        # add BatchNorm layer after the last conv layer
        self.sig = nn.Sigmoid()

        # define theta and phi for all operations except gaussian
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

            
    def forward(self, x, return_nl_map=False):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation

        theta_x = self.theta(x).view(batch_size, self.in_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / math.sqrt(self.in_channels)

        # contiguous here just allocates contiguous chunk of memory
        y = f_div_C.permute(0, 2, 1).contiguous()
        
        sig_y = self.sig(y)
        final_y = sig_y.view(batch_size, *x.size()[2:], *x.size()[2:])

        if return_nl_map:
            return final_y, sig_y
        else:
            return final_y

# if __name__ == '__main__':

#     img = torch.zeros(64, 256, 37, 37)
#     net = NLBlockND(in_channels=256)
#     out, feature_map = net.forward(img, True)
#     print(out.size())
#     print(feature_map.size())