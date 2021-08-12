"""
modified from PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
from utils import util
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    cache = dir.rstrip('/') + '.txt'
    if os.path.isfile(cache):
        print("Using filelist cached at %s" % cache)
        with open(cache) as f:
            images = [line.strip() for line in f]
        # patch up image list with new loading method
        if images[0].startswith(dir):
            print("Using image list from older version")
            image_list = []
            for image in images:
                image_list.append(image)
        else:
            print("Adding prefix to saved image list")
            image_list = []
            prefix = os.path.dirname(dir.rstrip('/'))
            for image in images:
                image_list.append(os.path.join(prefix, image))
        image_list = image_list[:min(max_dataset_size, len(image_list))]
        # image_list = random.sample(image_list, min(
        #     max_dataset_size, len(image_list)))
        return image_list
    print("Walking directory ...")
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    image_list = images[:min(max_dataset_size, len(images))]
    # image_list = random.sample(images, min(max_dataset_size, len(images)))
    with open(cache, 'w') as f:
        prefix = os.path.dirname(dir.rstrip('/')) + '/'
        for i in image_list:
            f.write('%s\n' % util.remove_prefix(i, prefix))

    new_data_list = []
    orig_vid = set([x.split('/')[-1].split('_')[0] for x in image_list])
    for i, vid_name in enumerate(orig_vid):
        vids = list(filter(lambda x: x.split('/')[-1].split('_')[0] == vid_name, image_list))
        selected_frame = random.sample(vids, 32)
        new_data_list += selected_frame

    return new_data_list


def make_multiple_dataset(dir, max_dataset_size=float("inf")):
    subdir = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    total_image_list = []
    last_dir, dir = dir.split(
        '/')[-2] + '/' + dir.split('/')[-1], '/'.join(dir.split('/')[:-2])
    print(dir)
    for sdir in subdir:
        curr_dir = dir + '/' + sdir + '/' + last_dir + '/'
        print(curr_dir)
        cache = curr_dir.rstrip('/') + '.txt'
        if os.path.isfile(cache):
            print("Using filelist cached at %s" % cache)
            with open(cache) as f:
                images = [line.strip() for line in f]
            # patch up image list with new loading method
            if images[0].startswith(curr_dir):
                print("Using image list from older version")
                image_list = []
                for image in images:
                    image_list.append(image)
            else:
                print("Adding prefix to saved image list")
                image_list = []
                prefix = os.path.dirname(curr_dir.rstrip('/'))
                for image in images:
                    image_list.append(os.path.join(prefix, image))
            image_list = random.sample(image_list, min(
                max_dataset_size, len(image_list)))
            total_image_list += image_list
        else:
            print("Walking directory ...")
            images = []
            assert os.path.isdir(
                curr_dir), '%s is not a valid directory' % curr_dir
            for root, _, fnames in sorted(os.walk(curr_dir, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
            # image_list = images[:min(max_dataset_size, len(images))]
            image_list = random.sample(
                images, min(max_dataset_size, len(images)))
            with open(cache, 'w') as f:
                prefix = os.path.dirname(curr_dir.rstrip('/')) + '/'
                for i in image_list:
                    f.write('%s\n' % util.remove_prefix(i, prefix))
            total_image_list += image_list
    return total_image_list


def make_multiple_dataset_real(dir, max_dataset_size=float("inf")):
    subdir = ['faces/celebahq/real-tfr-1024-resized128', 'faces/celebahq/real-tfr-1024-resized128',
              'faces/celebahq/real-tfr-1024-resized128', 'faceforensics_aligned/Deepfakes/original',
              'faceforensics_aligned/Face2Face/original', 'faceforensics_aligned/FaceSwap/original',
              'faceforensics_aligned/NeuralTextures/original']
    total_image_list = []
    last_dir, dir = dir.split('/')[-1], '/'.join(dir.split('/')[:-1])
    print(dir)
    for sdir in subdir:
        curr_dir = dir + '/' + sdir + '/' + last_dir + '/'
        print(curr_dir)
        cache = curr_dir.rstrip('/') + '.txt'
        if os.path.isfile(cache):
            print("Using filelist cached at %s" % cache)
            with open(cache) as f:
                images = [line.strip() for line in f]
            # patch up image list with new loading method
            if images[0].startswith(curr_dir):
                print("Using image list from older version")
                image_list = []
                for image in images:
                    image_list.append(image)
            else:
                print("Adding prefix to saved image list")
                image_list = []
                prefix = os.path.dirname(curr_dir.rstrip('/'))
                for image in images:
                    image_list.append(os.path.join(prefix, image))
            image_list = random.sample(image_list, min(
                max_dataset_size, len(image_list)))
            total_image_list += image_list
        else:
            print("Walking directory ...")
            images = []
            assert os.path.isdir(
                curr_dir), '%s is not a valid directory' % curr_dir
            for root, _, fnames in sorted(os.walk(curr_dir, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
            # image_list = images[:min(max_dataset_size, len(images))]
            image_list = random.sample(
                images, min(max_dataset_size, len(images)))
            with open(cache, 'w') as f:
                prefix = os.path.dirname(curr_dir.rstrip('/')) + '/'
                for i in image_list:
                    f.write('%s\n' % util.remove_prefix(i, prefix))
            total_image_list += image_list
    return total_image_list


def make_multiple_dataset_fake(dir, max_dataset_size=float("inf")):
    subdir = ['faces/celebahq/pgan-pretrained-128-png', 'faces/celebahq/sgan-pretrained-128-png',
              'faces/celebahq/glow-pretrained-128-png', 'faceforensics_aligned/Deepfakes/manipulated',
              'faceforensics_aligned/Face2Face/manipulated', 'faceforensics_aligned/FaceSwap/manipulated',
              'faceforensics_aligned/NeuralTextures/manipulated']
    total_image_list = []
    last_dir, dir = dir.split('/')[-1], '/'.join(dir.split('/')[:-1])
    print(dir)
    for sdir in subdir:
        curr_dir = dir + '/' + sdir + '/' + last_dir + '/'
        print(curr_dir)
        cache = curr_dir.rstrip('/') + '.txt'
        if os.path.isfile(cache):
            print("Using filelist cached at %s" % cache)
            with open(cache) as f:
                images = [line.strip() for line in f]
            # patch up image list with new loading method
            if images[0].startswith(curr_dir):
                print("Using image list from older version")
                image_list = []
                for image in images:
                    image_list.append(image)
            else:
                print("Adding prefix to saved image list")
                image_list = []
                prefix = os.path.dirname(curr_dir.rstrip('/'))
                for image in images:
                    image_list.append(os.path.join(prefix, image))
            image_list = random.sample(image_list, min(
                max_dataset_size, len(image_list)))
            total_image_list += image_list
        else:
            print("Walking directory ...")
            images = []
            assert os.path.isdir(
                curr_dir), '%s is not a valid directory' % curr_dir
            for root, _, fnames in sorted(os.walk(curr_dir, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
            # image_list = images[:min(max_dataset_size, len(images))]
            image_list = random.sample(
                images, min(max_dataset_size, len(images)))
            with open(cache, 'w') as f:
                prefix = os.path.dirname(curr_dir.rstrip('/')) + '/'
                for i in image_list:
                    f.write('%s\n' % util.remove_prefix(i, prefix))
            total_image_list += image_list
    return total_image_list


def make_CNNDetection_dataset(dir, max_dataset_size=float("inf"), mode='real'):
    classes = os.listdir(dir)
    total_image_list = []
    total_class_list = []
    print(dir)
    if mode == 'real':
        sdir = '0_real'
    elif mode == 'fake':
        sdir = '1_fake'
    for cls in classes:
        curr_dir = dir + '/' + cls + '/' + sdir
        print(curr_dir)
        cache = curr_dir.rstrip('/') + '.txt'
        if os.path.isfile(cache):
            print("Using filelist cached at %s" % cache)
            with open(cache) as f:
                images = [line.strip() for line in f]
            # patch up image list with new loading method
            if images[0].startswith(curr_dir):
                print("Using image list from older version")
                image_list = []
                class_list = []
                for image in images:
                    image_list.append(image)
                    class_list.append(cls)
            else:
                print("Adding prefix to saved image list")
                image_list = []
                class_list = []
                prefix = os.path.dirname(curr_dir.rstrip('/'))
                for image in images:
                    image_list.append(os.path.join(prefix, image))
                    class_list.append(cls)
            total_image_list += image_list
            total_class_list += class_list
        else:
            print("Walking directory ...")
            images = []
            class_list = []
            assert os.path.isdir(
                curr_dir), '%s is not a valid directory' % curr_dir
            for root, _, fnames in sorted(os.walk(curr_dir, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
                        class_list.append(cls)
            image_list = images
            with open(cache, 'w') as f:
                prefix = os.path.dirname(curr_dir.rstrip('/')) + '/'
                for i in image_list:
                    f.write('%s\n' % util.remove_prefix(i, prefix))
            total_image_list += image_list
            total_class_list += class_list
    return total_image_list, total_class_list



def default_loader(path):
    return Image.open(path).convert('RGB')
