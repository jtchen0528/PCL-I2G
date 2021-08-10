'''

Reference from @author Zhuolin Fu

'''

import argparse, sys, os
from os.path import basename, splitext
# from PIL import Image
from functools import partial

from skimage.transform import PiecewiseAffineTransform, warp
from random import uniform
import numpy as np
import cv2
from tqdm import tqdm

SCALE, SHAKE_H = 0.8, 0.2
SAVE_BLEND, OUT_PATH = True, '/nas/hjr/tempBlended_faceswap'
COLOR_TRANSFER = 'faceswap'
BLEND_TYPE = 'faceswap'

if COLOR_TRANSFER == 'faceswap':
    from data.processing.blend_utils.color_transfer_faceswap import color_transfer
else:
    from data.processing.blend_utils.color_transfer import color_transfer
from data.processing.blend_utils.utils import files, FACIAL_LANDMARKS_IDXS, shape_to_np


def cv_loader(path, gray=False):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except IOError:
        print('Cannot load image ' + path)


def get_landmarks(detector, predictor, rgb):
    # first get bounding box (dlib.rectangle class) of face.
    boxes = []
    if detector:
        boxes = detector(rgb, 1)
        # pdb.set_trace()
    for box in boxes:
        landmarks = shape_to_np(predictor(rgb, box=box))
        break
    else:
        return None
    return landmarks.astype(np.int32)


def find_one_neighbor(detector, predictor, srcPath, srcLms, faceDatabase, threshold):
    import dlib
    for face in faceDatabase:
        rgb = dlib.load_rgb_image(face)
        landmarks = get_landmarks(detector, predictor, rgb)
        if landmarks is None:
            continue
        dist = distance(srcLms, landmarks)
        if dist < threshold and basename(face).split('_')[0] != basename(srcPath).split('_')[0]:
            return rgb
    return None


def get_roi(warped):
    '''返回 warped 区域的 roi 边框
    warped: (h, w, c), float64, [0, 1]
    return: left, up, right, bot.
    '''
    height, width = warped.shape[:2]
    left, up, right, bot = 0, 0, width, height
    gray = warped[:, :, 0]
    rowHistogram, colHistogram = gray.sum(axis=0), gray.sum(axis=1)
    for i in range(width):
        if rowHistogram[i] != 0:
            left = i
            break
    for i in range(width-1, -1, -1):
        if rowHistogram[i] != 0:
            right = i
            break
    for i in range(height):
        if colHistogram[i] != 0:
            up = i
            break
    for i in range(height-1, -1, -1):
        if colHistogram[i] != 0:
            bot = i
            break
    ''' Old style Implementeation. Maybe something is wrong.
    for i, num in enumerate(rowHistogram):
        if left == 0 and num !=0:
            left = i
        if i > 0 and rowHistogram[i-1]>0 and num==0 and right == 0:
            right = i
    for i, num in enumerate(colHistogram):
        if up == 0 and num !=0:
            up = i
        if i > 0 and colHistogram[i-1]>0 and num==0 and bot == 0:
            bot = i
    '''
    return left, up, right, bot


def forge(srcRgb, targetRgb, mask):

    return (mask * targetRgb + (1 - mask) * srcRgb).astype(np.uint8)

def get_bounding(mask):
    bounding = 4 * mask * (1 - mask)
    # print(type(mask), mask.shape, mask.dtype)
    # bounding = np.zeros((mask.shape[1], mask.shape[0], 3))
    # for i in range(mask.shape[1]):
    #     for j in range(mask.shape[0]):
    #         bounding[i, j] = mask[i, j] * (1 - mask[i, j]) * 4 # 处理每个像素点
    return bounding


def convex_hull(size, points, fillColor=(255,)*3):
    mask = np.zeros(size, dtype=np.uint8) # mask has the same depth as input image
    points = cv2.convexHull(np.array(points))
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, fillColor)
    return mask


def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h, nrows).astype(np.int32)
    cols = np.linspace(0, w, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return anchors, deformed.astype(np.int32)


def linear_deform(warped, scale=0.5, shake_h=0.2, random=True):
    """缩放+高度抖动

    params:
        warped {np.ndarray} -- float mask of areas for transfer.
        scale {float}  -- random minimum scale
            1.0 for keep original scale, 0.0 for one pixel
        shake_h {float} -- random minimum shake for height.
            1.0 for no shake, 0.01 for shake from bottom
    return:
        deformed {np.ndarray} -- float mask.
    """
    if shake_h == 0.0:
        shake_h = 0.001
    h, w, _ = warped.shape
    deformed = np.zeros_like(warped)
    # cv2.imwrite('warped.jpg', warped*255)
    scaleRandom, shakeRandom = scale, shake_h
    if random:
        # randPair = np.random.rand(2)
        # scaleRandom = 1-randPair[0]*scale  # [scale, 1]
        # shakeRandom = randPair[1]*shake_h  # [0， shake_h]
        scaleRandom = uniform(min(1, scale), 1.0)
        shakeRandom = uniform(min(1, shake_h), 1.0)
    # print(scaleRandom, shakeRandom)
    hScale, wScale = int(h*scaleRandom), int(w*scaleRandom)
    warped = cv2.resize(warped, (wScale, hScale))
    hPlus = int((1-shakeRandom)*(h-hScale)//2)
    hNew, wNew = int((h-hScale)//2), int((w-wScale)//2)
    hNew += hPlus
    deformed[hNew: hNew+hScale, wNew: wNew+wScale, :] += warped
    # cv2.imwrite('deformed.jpg', deformed*255)
    return deformed


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    '''  Return 0-1 range
    '''
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped


def distance(lms1, lms2):
    return np.linalg.norm(lms1 - lms2)  # 两landmarks的二范数 = 欧几里得距离

############# Blend helpers ###############

import json
import os.path as osp
from random import sample, uniform

def getRelative(path):
    """
    xxx/000.mp4/0.jpg -> 000.mp4/0.jpg
    """
    path, name = osp.split(path)
    _, id = osp.split(path)
    relativePath = osp.join(id, name)
    return relativePath

def getName(path):
    """
    000.mp4/0.jpg -> 004.mp4_0
    """
    '_'.join(osp.split(path)).rstrip('.jpg')

class kernelSampler:
    """ 支持 int 类型，或者 list(int), tuple(int)
    """
    def __init__(self, kernel):
        if not isinstance(kernel, (list, tuple, int)):
            raise NotImplementedError(kernel)
        self.kernel = kernel
        if not isinstance(kernel, int):
            assert len(kernel) == 2
            self.kernel = []
            for i in range(kernel[0], kernel[1]):
                if i % 2 == 1:
                    self.kernel.append(i)

    def __call__(self):
        if isinstance(self.kernel, int):
            return self.kernel
        else:
            return sample(self.kernel, k=1)[0]


class sigmaSampler:
    """ 支持 float 类型，或者 list(a, b), tuple(a, b)
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self):
        if isinstance(self.sigma, (int, float)):
            return self.kernel
        else:
            return uniform(self.sigma[0], self.sigma[1])


class Blender:
    '''贴合器
    '''
    def __init__(self, ldmPath, dataPath, topk=100, selectNum=1, \
            gaussianKernel=5, gaussianSigma=7, loader='cv',
            pixel_aug=None, spatial_aug=None, aug_at_load=False
        ):
        # 格式读取、转化。
        self.relativePaths, lms = [], []
        self.dataPath = dataPath
        self.topk = topk
        self.selectNum = selectNum
        self.gaussianKernel = gaussianKernel

        if ldmPath:
            with open(ldmPath, 'r') as f:
                relatPath_lms = json.load(f)
                for path, lm in relatPath_lms:
                    if lm is None:  continue
                    path = getRelative(path)
                    self.relativePaths.append(path)
                    lms.append(lm)
            self.lms = np.array(lms)  # 用于计算相似度矩阵
            N = self.lms.shape[0]
            self.lms = np.reshape(self.lms, (N, -1))
            print(self.lms.shape)  # (N, 136)
            self.kSampler = kernelSampler(gaussianKernel)
            self.sSampler = sigmaSampler(gaussianSigma)

        if loader == 'cv':
            self.loader_fn = cv_loader
        else:
            raise NotImplementedError(loader)

        self.pixel_aug = pixel_aug
        self.spatial_aug = spatial_aug

        self.aug_at_load = aug_at_load
        print('[Blender]: aug_at_load:', self.aug_at_load)

    def __len__(self):
        return self.lms.shape[0]

    def img_loader(self, i, gray=False, do_aug=False):
        """ default: not to aug at image reading
        """
        if isinstance(i, str):
            imgPath = i
        else:
            i = int(i)
            path = self.relativePaths[i]
            imgPath = osp.join(self.dataPath, path)
        
        img = self.loader_fn(imgPath, gray=gray)
        if img is None:
            print('Error imgPath: ', imgPath)
        # 检查并做 augmentation
        if do_aug and self.pixel_aug:
            img = self.pixel_aug(img)
        return img

    def core_xray(self, imgPair, warped):
        # 高斯模糊
        # blured = cv2.GaussianBlur(warped, (self.gaussianKernel, self.gaussianKernel), 3)
        ksize, sigma = self.kSampler(), self.sSampler()
        # print(ksize, sigma)
        blured = cv2.GaussianBlur(warped, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        # 颜色矫正，迁移高斯模糊后blured mask区域的颜色，并对颜色纠正的blured区域人脸+原背景作为融合图片
        left, up, right, bot = get_roi(blured)  # 获取 warped 区域
        
        # src 取中间部分采集色彩（效果感觉差了很多，还是得改基于 mask 的色彩迁移）
        # h, w = bot-up, right-left
        # src = (imgPair[0][up+h//4:bot-h//4,left+w//4:right-w//4,:]).astype(np.uint8)
        src = (imgPair[0][up:bot,left:right,:]).astype(np.uint8)  # 这里命名有误。src 应当是前景
        tgt = (imgPair[1][up:bot,left:right,:]).astype(np.uint8)  # 这里命名有误。tgt 应当是背景
        # 基于 mask 的色彩迁移
        targetBgrT = color_transfer(src, tgt, preserve_paper=False, mask=blured[up:bot,left:right,0]!=0)
        # pdb.set_trace()
        targetBgr_T = imgPair[1] * 1  # 开辟新内存空间
        targetBgr_T[up:bot,left:right,:] = targetBgrT  # 将色彩迁移的部分转移到原图片
        # 融合
        resultantFace = forge(imgPair[0], targetBgr_T, blured)  # forged face
        # 混合边界
        resultantBounding = get_bounding(blured)
        return resultantFace, resultantBounding

    def core_alpha(self, imgPair, warped, featherAmount=0.2):
        # from FF++ source code faceSwap
        dst, src = imgPair[0], imgPair[1]
        mask = warped

        src = color_transfer(dst, src, preserve_paper=False, mask=mask)

        #indeksy nie czarnych pikseli maski
        maskIndices = np.where(mask != 0)
        #te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
        maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
        faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
        featherAmount = featherAmount * np.max(faceSize)

        hull = cv2.convexHull(maskPts)
        dists = np.zeros(maskPts.shape[0])
        for i in range(maskPts.shape[0]):
            dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

        weights = np.clip(dists / featherAmount, 0, 1)
        # import pdb; pdb.set_trace()

        composedImg = np.copy(dst)
        composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]
        alpha_mask = np.zeros_like(mask)
        alpha_mask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * 1
        xray = get_bounding(alpha_mask)
        return composedImg, xray

    def core(self, i, j):
        '''贴合：用 i 的背景，接纳 j 的前景（j 攻击 i）
        '''
        imgPair = [self.img_loader(k, do_aug=self.aug_at_load) for k in (i, j)]
        lms = [self.lms[i].reshape(-1,2) for k in (i, j)]
        
        hullMask = convex_hull(imgPair[0].shape, lms[0])  # todo: shrink mask.
        # 只对 mask 部分 random deform
        left, up, right, bot = get_roi(hullMask)
        left, up, right, bot = (left+0)//2, (up+0)//2, (right+hullMask.shape[1])//2, (bot+hullMask.shape[0])//2
        centerHullMask = hullMask[up:bot, left:right, :]
        anchors, deformedAnchors = random_deform(centerHullMask.shape[:2], 4, 4)  # todo 方法不够理想
        warpedMask = piecewise_affine_transform(centerHullMask, anchors, deformedAnchors)
        # 伪造区域随机化：进一步缩放+平移抖动
        warpedMask = linear_deform(warpedMask, scale=SCALE, shake_h=SHAKE_H, random=True)
        # 将 warped 区域限制在人脸范围内，避免背景的影响
        # warpedMask *= (centerHullMask / centerHullMask.max())
        # 还原
        warped = np.zeros_like(hullMask, dtype=warpedMask.dtype)
        warped[up:bot, left:right, :] = warpedMask
        # pdb.set_trace()
        if BLEND_TYPE == 'faceswap':
            resultantFace, resultantBounding = self.core_alpha(imgPair, warped)
        else:
            resultantFace, resultantBounding = self.core_xray(imgPair, warped)

        return resultantFace, resultantBounding

    def search(self, idx):
        ''' 保证不重复
        '''
        topk = min(len(self.lms)-1, self.topk)
        selectNum = min(self.selectNum, topk)
        pivot = self.lms[idx]
        subs = self.lms-pivot
        scores = (subs**2).sum(-1)  # l2 距离
        idxes = np.argpartition(scores, topk)[:topk]  # topK
        # 去重
        # 要忽略的集合
        ignoring = [idx]
        ignoring = [i for i in range(idx-100, idx+100)]  # 前后的100个都不要了
        filteredIndexes = [i for i in idxes if i not in ignoring]
        # pdb.set_trace()
        outs = sample(filteredIndexes, k=selectNum)  # 对 idx 去重
        # pdb.set_trace()
        return outs

    def blend_i(self, i, get_path=False):
        """blend: default do aug
        """
        i_path = self.relativePaths[i]
        # import time
        # start = time.clock()
        js = self.search(i) # 0.12s
        # time_search = time.clock() - start
        # print('TIME search:', time_search) 
        for j in js:
            j_path = self.relativePaths[j]
            # start = time.clock()
            blended, label = self.core(i, j)  # 0.64s
            # time_core = time.clock() - start
            # print('TIME core:', time_core)
            if self.pixel_aug:
                blended = self.pixel_aug(blended)
            if SAVE_BLEND:
                name = '{}_{}'.format(i, j)  # j attack i
                status = 0
                blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                status += cv2.imwrite(osp.join(OUT_PATH, name+'.png'), blended_bgr)
                status += cv2.imwrite(osp.join(OUT_PATH, name+'_label'+'.png'), label*255)
                assert status == 2, 'Error: image saving failed: {}/{}'.format(OUT_PATH, name)
            if get_path:
                yield blended, label, i_path, j_path  # generator
            else:
                yield blended, label


if __name__ == '__main__':
    blender = Blender(ldmPath='x', dataPath='x', topk=1, selectNum=1, gaussianKernel=[1, 3], gaussianSigma=[1, 3])
