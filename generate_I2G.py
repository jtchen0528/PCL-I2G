import numpy as np
import os
from numpy.core.numeric import Inf
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.I2G_dataset import I2GDataset
import argparse
import sys

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

parser = argparse.ArgumentParser()
parser.add_argument("--real_im_path", type=str, required=True, help="path of the real image")
parser.add_argument("--batch_size", type=int, default=512, help="batch size to generate images")
parser.add_argument("--out_size", type=int, default=256, help="image output size")
parser.add_argument("--output_dir", type=str, required=True, help="path to store output images")
parser.add_argument("--output_max", type=int, default=140000, help="max output image number")
args = parser.parse_args()

opt = {
    'real_im_path': args.real_im_path,
    'nThreads': 4,
    'batch_size': args.batch_size,
    'loadSize': args.out_size,
    'fineSize': args.out_size,
    'output_dir': args.output_dir
}
opt = Struct(**opt)

os.makedirs(os.path.join(opt.output_dir, 'real', 'face'), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, 'real', 'mask'), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, 'fake', 'face'), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, 'fake', 'mask'), exist_ok=True)
count_real = 0
count_fake = 0

while count_real <= args.output_max or count_fake <= args.output_max:
    dset = I2GDataset(opt, os.path.join(opt.real_im_path), orig_transform=True)
    dset.get32frames()
    dl = DataLoader(dset, batch_size=opt.batch_size,
                    num_workers=opt.nThreads, pin_memory=False,
                    shuffle=False)
    total_batches = len(dl)
    print(total_batches)
    for i, ims in enumerate(dl):
        if i % 20 == 0:
            print('finished: {}/{}'.format(i, total_batches))
        for j in range(len(ims['img'])):
            img_save = transforms.ToPILImage()(ims['img'][j]).convert("RGB")
            mask_save = transforms.ToPILImage()(ims['mask'][j]).convert("L")
            if ims['label'][j] == 1:
                img_save.save(os.path.join(opt.output_dir, 'real', 'face', '%d.png') % count_real)
                mask_save.save(os.path.join(opt.output_dir, 'real', 'mask', '%d.png') % count_real)
                count_real += 1
            else:
                img_save.save(os.path.join(opt.output_dir, 'fake', 'face', '%d.png') % count_fake)
                mask_save.save(os.path.join(opt.output_dir, 'fake', 'mask', '%d.png') % count_fake)
                count_fake += 1
        # if count_real >= 100:
        #     sys.exit('exit')
