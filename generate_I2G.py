import numpy as np
import os
from numpy.core.numeric import Inf
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.I2G_dataset_2 import I2GDataset
from PIL import Image
from collections import namedtuple

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

opt = {
    'max_dataset_size': Inf,
    'real_im_path': '/scratch3/users/clairelai/faceforensics_aligned/Deepfakes/original',
    'nThreads': 4,
    'batch_size': 512,
    'loadSize': 256,
    'fineSize': 256
}
opt = Struct(**opt)

dset = I2GDataset(opt, os.path.join(opt.real_im_path, 'train'))
    # halves batch size since each batch returns both real and fake ims

dset.get32frames()

dl = DataLoader(dset, batch_size=opt.batch_size,
                num_workers=opt.nThreads, pin_memory=False,
                shuffle=True)
total_batches = len(dl)
os.makedirs('I2G_dataset', exist_ok=True)
os.makedirs('I2G_dataset/real', exist_ok=True)
os.makedirs('I2G_dataset/real/face', exist_ok=True)
os.makedirs('I2G_dataset/real/mask', exist_ok=True)
os.makedirs('I2G_dataset/fake', exist_ok=True)
os.makedirs('I2G_dataset/fake/face', exist_ok=True)
os.makedirs('I2G_dataset/fake/mask', exist_ok=True)
count = 0
for i, ims in enumerate(dl):
    if i % 20 == 0:
        print('finished: {}/{}%'.format(i, total_batches))
    for j in range(len(ims['img'])):
        img_save = transforms.ToPILImage()(ims['img'][j]).convert("RGB")
        mask_save = transforms.ToPILImage()(ims['mask'][j]).convert("L")
        if ims['label'][j] == 1:
            img_save.save('I2G_dataset/real/face/%d.png' % count)
            mask_save.save('I2G_dataset/real/mask/%d.png' % count)
        else:
            img_save.save('I2G_dataset/fake/face/%d.png' % count)
            mask_save.save('I2G_dataset/fake/mask/%d.png' % count)
        count += 1
