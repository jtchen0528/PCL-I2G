
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch.transforms import ToTensor


def simple_transform():
    t = Compose([
        Resize(256, 256)
    ])
    return t


def strong_aug_pixel(p=.5):
    print('[DATA]: strong aug pixel')

    from albumentations import (
    # HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, MultiplicativeNoise,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, JpegCompression, CLAHE)

    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        OneOf([
            MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True),
            JpegCompression(quality_lower=39, quality_upper=80)
        ], p=0.2),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def pixel_aug(p=.5):
    print('[DATA]: pixel aug')

    from albumentations import JpegCompression, Blur, Downscale, CLAHE, HueSaturationValue, \
        RandomBrightnessContrast, IAAAdditiveGaussianNoise, GaussNoise, GaussianBlur, MedianBlur, MotionBlur, \
        Compose, OneOf
    from random import sample, randint, uniform

    return Compose([
        # Jpeg Compression
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=99, p=1)
        ], p=0.2),
        # Gaussian Noise
        OneOf([
            IAAAdditiveGaussianNoise(loc=randint(1, 9), p=1),
            GaussNoise(mean=uniform(0, 10.0), p=1),
        ], p=0.3),
        # Blur
        OneOf([
            GaussianBlur(blur_limit=15, p=1),
            MotionBlur(blur_limit=19, p=1),
            Downscale(scale_min=0.3, scale_max=0.99, p=1),
            Blur(blur_limit=15, p=1),
            MedianBlur(blur_limit=9, p=1)
        ], p=0.4),
        # Color
        OneOf([
            CLAHE(clip_limit=4.0, p=1),
            HueSaturationValue(p=1),
            RandomBrightnessContrast(p=1),
        ], p=0.1)
    ], p=p)


def spatial_aug(p=0.5):
    print('[DATA] spatial aug')
    from albumentations import (
        GridDropout, RandomResizedCrop, Rotate, HorizontalFlip, Compose)
    
    aug = Compose([
            # RandomSizedCrop(min_max_height=(randSize, randSize), height=256, width=256, p=0.5),
            GridDropout(holes_number_x=3, holes_number_y=3, random_offset=True, p=0.5),
            RandomResizedCrop(256, 256, scale=(0.7,1.0), p=1.0),
            HorizontalFlip(p=0.5),
            Rotate(limit=90, p=0.5),
        ], p=p)

    return aug



def pixel_aug_mild(p=.5):
    print('[DATA]: pixel aug mild')

    from albumentations import JpegCompression, Blur, Downscale, CLAHE, HueSaturationValue, \
        RandomBrightnessContrast, IAAAdditiveGaussianNoise, GaussNoise, GaussianBlur, MedianBlur, MotionBlur, \
        Compose, OneOf
    from random import sample, randint, uniform

    return Compose([
        # Jpeg Compression
        OneOf([
            JpegCompression(quality_lower=60, quality_upper=99, p=1)
        ], p=0.2),
        # Gaussian Noise
        OneOf([
            IAAAdditiveGaussianNoise(loc=randint(1, 5), p=1),
            GaussNoise(mean=uniform(0, 5.0), p=1),
        ], p=0.3),
        # Blur
        OneOf([
            GaussianBlur(blur_limit=7, p=1),
            MotionBlur(blur_limit=9, p=1),
            Downscale(scale_min=0.6, scale_max=0.99, p=1),
            Blur(blur_limit=7, p=1),
            MedianBlur(blur_limit=3, p=1)
        ], p=0.4),
        # Color
        OneOf([
            CLAHE(clip_limit=2.0, p=1),
            HueSaturationValue(p=1),
            RandomBrightnessContrast(p=1),
        ], p=0.1)
    ], p=p)



class Augmentator:
    def __init__(self, augment_fn=''):
        if augment_fn == 'pixel_aug':
            self.augment_fn = pixel_aug()
        elif augment_fn == 'simple':
            self.augment_fn = simple_transform()
        elif augment_fn == 'pixel_mild':
            self.augment_fn = pixel_aug_mild()
        elif augment_fn == 'spatial':
            self.augment_fn = spatial_aug()
        else:
            raise NotImplementedError(augment_fn)
        
    def __call__(self, img, mask=None):
        if mask is None:
            return self.augment_fn(image = img)['image']
        else:
            augmented = self.augment_fn(image=img, mask=mask)
            # import pdb; pdb.set_trace()
            return augmented['image'], augmented['mask']


def data_transform(size=256, normalize=True):
    if normalize:
        t = Compose([
            Resize(size, size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor()
        ])
    else:
        t = Compose([
            Resize(size, size),
            ToTensor()
        ])
    return t
