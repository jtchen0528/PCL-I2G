import torch
import torch.nn as nn
from IPython import embed
from . import netutils

def modify_commandline_options(parser):
    opt, _ = parser.parse_known_args()
    if 'xception' in opt.which_model_netD:
        parser.set_defaults(loadSize=333, fineSize=299)
    elif 'resnet' in opt.which_model_netD:
        parser.set_defaults(loadSize=256, fineSize=224)
    else:
        raise NotImplementedError

def define_D(which_model_netD, init_type, gpu_ids=[]):
    if 'resnet' in which_model_netD:
        from torchvision.models import resnet
        model = getattr(resnet, which_model_netD)
        netD = model(pretrained=False, num_classes=2)
    elif 'xception' in which_model_netD:
        from . import xception
        netD = xception.xception(num_classes=2)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)

def define_patch_D(which_model_netD, init_type, gpu_ids=[]):
    if which_model_netD.startswith('resnet'):
        # e.g. which_model_netD = resnet18_layer1
        from . import customnet
        splits = which_model_netD.split('_')
        depth = int(splits[0][6:])
        layer = splits[1]
        if len(splits) == 2:
            netD = customnet.make_patch_resnet(depth, layer)
        else:
            extra_output = [i.replace('extra', 'layer')
                            for i in splits[2:]]
            netD = customnet.make_patch_resnet(depth, layer, extra_output=extra_output)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('widenet'):
        # e.g. which_model_netD = widenet_kw7_d1
        splits = which_model_netD.split('_')
        kernel_size = int(splits[1][2:])
        dilation = int(splits[2][1:])
        netD = WideNet(kernel_size, dilation)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('xception'):
        # e.g. which_model_netD = xceptionnet_block2
        from . import customnet
        splits = which_model_netD.split('_')
        layer = splits[1]
        if len(splits) == 2:
            netD = customnet.make_patch_xceptionnet(layer)
        else:
            extra_output = [i.replace('extra', 'block')
                            for i in splits[2:]]
            netD = customnet.make_patch_xceptionnet(
                layername=layer, extra_output=extra_output)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('longxception'):
        from . import customnet
        netD = customnet.make_xceptionnet_long()
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

def define_PCL(which_model_netD, init_type, gpu_ids=[], input_size=128):
    if which_model_netD.startswith('resnet'):
        # e.g. which_model_netD = resnet18_layer1
        from . import customnet
        backbone = which_model_netD.split('_')[0]
        layer = which_model_netD.split('_')[1]
        netPCL, out_ch = customnet.make_pcl(backbone=backbone, layername=layer, input_size=input_size)
        return netutils.init_net(netPCL, init_type, gpu_ids=gpu_ids), out_ch
    elif which_model_netD.startswith('widenet'):
        # e.g. which_model_netD = widenet_kw7_d1
        splits = which_model_netD.split('_')
        kernel_size = int(splits[1][2:])
        dilation = int(splits[2][1:])
        netD = WideNet(kernel_size, dilation)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('xception'):
        # e.g. which_model_netD = xceptionnet_block2
        from . import customnet
        splits = which_model_netD.split('_')
        backbone = splits[0]
        layer = splits[1]
        netPCL, out_ch = customnet.make_pcl(backbone=backbone, layername=layer, input_size=input_size)
        return netutils.init_net(netPCL, init_type, gpu_ids=gpu_ids), out_ch

    elif which_model_netD.startswith('longxception'):
        from . import customnet
        netD = customnet.make_xceptionnet_long()
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

class WideNet(nn.Module):
    # a shallow network based off initial layers of resnet with 
    # a few 1x1 conv layers added on
    def __init__(self, kernel_size=7, dilation=1):
        super().__init__()
        sequence = [
            nn.Conv2d(3, 256, kernel_size=kernel_size, dilation=dilation,
                      stride=2, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # linear layers
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

