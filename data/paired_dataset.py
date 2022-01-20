import os.path
import torch.utils.data as data
from .dataset_util import make_dataset
from PIL import Image
import numpy as np
import torch
from . import transforms
import random

class PairedDataset(data.Dataset):
    """A dataset class for paired images
    e.g. corresponding real and manipulated images
    """

    def __init__(self, opt, im_path_real, im_path_fake, is_val=False, with_mask=False):
        """Initialize this dataset class.

        Parameters:
            opt -- experiment options
            im_path_real -- path to folder of real images
            im_path_fake -- path to folder of fake images
            is_val -- is this training or validation? used to determine
            transform
        """
        super().__init__()
        self.dir_real = im_path_real
        self.dir_fake = im_path_fake

        # if pairs are named in the same order 
        # e.g. real/train/face1.png, real/train/face2.png ...
        #      fake/train/face1.png, fake/train/face2.png ...
        # then this will align them in a batch unless
        # --no_serial_batches is specified
        self.with_mask = with_mask 

        if self.with_mask:
            self.real_paths = sorted([os.path.join(self.dir_real, im) for im in os.listdir(self.dir_real)])
            self.fake_paths = sorted([os.path.join(self.dir_fake, im) for im in os.listdir(self.dir_fake)])
        else:
            self.real_paths = sorted(make_dataset(self.dir_real,
                                                opt.max_dataset_size))
            self.fake_paths = sorted(make_dataset(self.dir_fake,
                                                opt.max_dataset_size))

        self.real_size = len(self.real_paths)
        self.fake_size = len(self.fake_paths)
        self.transform = transforms.get_transform(opt, for_val=is_val)

        if self.with_mask:
            self.real_mask_paths = sorted([os.path.join(self.dir_real.replace('face', 'mask'), im) for im in os.listdir(self.dir_real.replace('face', 'mask'))])
            self.fake_mask_paths = sorted([os.path.join(self.dir_fake.replace('face', 'mask'), im) for im in os.listdir(self.dir_fake.replace('face', 'mask'))])
            self.orig_transform = transforms.get_mask_transform(opt, for_val=is_val)
            self.real_mask_size = len(self.real_mask_paths)
            self.fake_mask_size = len(self.fake_mask_paths)
            assert(self.real_mask_size == self.real_size)
            assert(self.fake_mask_size == self.fake_size)

        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # read a image given a random integer index
        real_path = self.real_paths[index % self.real_size]  # make sure index is within then range
        # real_class = self.real_class[index % self.real_size]  # make sure index is within then range
        # fake_class = self.fake_class[index % self.fake_size]  # make sure index is within then range

        fake_path = self.fake_paths[index % self.fake_size]
        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')

        # apply image transformation
        real = self.transform(real_img)
        fake = self.transform(fake_img)

        if self.with_mask:
            real_mask_path = self.real_mask_paths[index % self.real_mask_size]
            fake_mask_path = self.fake_mask_paths[index % self.fake_mask_size]
            # apply image transformation
            real_mask = Image.open(real_mask_path).convert('L')
            fake_mask = Image.open(fake_mask_path).convert('L')
            real_mask = self.orig_transform(real_mask)
            fake_mask = self.orig_transform(fake_mask)
            
        if self.with_mask:
            return {'manipulated': fake,
                    'original': real,
                    'path_manipulated': fake_path,
                    'path_original': real_path,
                    'mask_original': real_mask,
                    'mask_manipulated': fake_mask,
                }
        else:
            return {'manipulated': fake,
                    'original': real,
                    'path_manipulated': fake_path,
                    'path_original': real_path,
                }

    def __len__(self):
        return max(self.real_size, self.fake_size)

class UnpairedDataset(data.Dataset):
    """A dataset class for loading images within a single folder
    """
    def __init__(self, opt, im_path, is_val=False):
        """Initialize this dataset class.

        Parameters:
            opt -- experiment options
            im_path -- path to folder of images
            is_val -- is this training or validation? used to determine
            transform
        """
        super().__init__()
        self.dir = im_path
        self.paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        self.size = len(self.paths)
        assert(self.size > 0)
        self.transform = transforms.get_transform(opt, for_val=is_val)
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # read a image given a random integer index
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        # apply image transformation
        img = self.transform(img)

        return {'img': img,
                'path': path
               }

    def __len__(self):
        return self.size

