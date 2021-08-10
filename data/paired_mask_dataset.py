import os.path
import torch.utils.data as data
from .dataset_util import make_dataset
from PIL import Image
import numpy as np
import torch
from . import transforms
import random
from data.processing.find_faces import find_face_cvhull
import cv2
import elasticdeform

class PairedMaskDataset(data.Dataset):
    """A dataset class for paired images
    e.g. corresponding real and manipulated images
    """

    def __init__(self, opt, im_path_real, im_path_fake, is_val=False, mode='single'):
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
        # then this will align them in a batch
        
        real_paths = make_dataset(self.dir_real)
        fake_paths = make_dataset(self.dir_fake)

        assert(len(real_paths) == len(fake_paths))
        if len(real_paths) > opt.max_dataset_size:
            random_indices = sorted(np.random.choice(len(real_paths), opt.max_dataset_size).tolist())
            real_paths = np.take(real_paths, random_indices)
            fake_paths = np.take(fake_paths, random_indices)

        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.real_size = len(self.real_paths)
        self.fake_size = len(self.fake_paths)
        self.transform = transforms.get_transform(opt, for_val=is_val)
        self.mask_transform = transforms.get_mask_transform(opt, for_val=is_val)
        self.opt = opt
        self.last_mask = np.ones((1, 1, 2))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # read a image given a random integer index
        # make sure index is within then range
        real_path = self.real_paths[index % self.real_size]
        fake_path = self.fake_paths[index % self.fake_size]
        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')
        # real_img.save("{}_real.png".format(index))
        real = self.transform(real_img)
        fake = self.transform(fake_img)

        fake_img_hull = find_face_cvhull(np.array(real_img))
        if fake_img_hull is None:
            fake_img_hull = self.last_mask
        else:
            self.last_mask = fake_img_hull
        H, W, C = np.array(real_img).shape

        real_mask = torch.ones([H, W])
        real_mask_img = Image.fromarray(np.uint8(real_mask * 255) , 'L')
        fake_mask = np.zeros([H, W, 1])
        cv2.fillPoly(fake_mask, [fake_img_hull], [1])
        fake_mask_deformed = elasticdeform.deform_random_grid(fake_mask[:, :, 0], sigma=0.01, points=4)
        fake_mask_deformed_blurred = cv2.GaussianBlur(fake_mask_deformed, (15, 15), 5)
        fake_mask_deformed_blurred = 1 - fake_mask_deformed_blurred
        fake_mask_img = Image.fromarray(np.uint8(fake_mask_deformed_blurred * 255) , 'L')
        # fake_mask_img.save("{}_mask.png".format(index))

        real_mask = self.mask_transform(real_mask_img)
        fake_mask = self.mask_transform(fake_mask_img)

        # real_img.save("masks/{}_real.png".format(index))
        # fake_img.save("masks/{}_fake.png".format(index))
        # real_mask_img.save("masks/{}_real_mask.png".format(index))
        # fake_mask_img.save("masks/{}_fake_mask.png".format(index))

        return {'manipulated': fake,
                'original': real,
                'path_manipulated': fake_path,
                'path_original': real_path,
                'mask_original': real_mask,
                'mask_manipulated': fake_mask,
        }

    def __len__(self):
        return max(self.real_size, self.fake_size)