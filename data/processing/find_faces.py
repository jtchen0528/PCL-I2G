from typing_extensions import final
from PIL import Image
import dlib
import cv2
import scipy.ndimage
import numpy as np
import torch

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rot90(v):
    return np.array([-v[1], v[0]])
    
def find_face_cvhull(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)

    hull = cv2.convexHull(shape)
    return hull

def find_face_landmark(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)

    return shape

class Masks4D(object):
    def __call__(self, masks):

        first_w = True
        first_h = True
        first_c = True

        for k, mask in enumerate(masks):
            h, w = mask.shape
            real_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask, 0), 0), 0)
            # fake_mask = torch.unsqueeze(torch.unsqueeze(1 - mask, 0), 0)
            for i, mask_h in enumerate(mask):
                for j, mask_w in enumerate(mask_h):
                    curr_mask = 1 - torch.abs(mask_w - real_mask)
                    if first_w:
                        total_mask_w = real_mask
                        first_w = False
                    else:
                        total_mask_w = torch.cat((total_mask_w, curr_mask), dim=2)
                if first_h:
                    total_mask_h = total_mask_w
                    first_h = False
                else:
                    total_mask_h = torch.cat((total_mask_h, total_mask_w), dim = 1)
                first_w = True
            if first_c:
                total_mask_c = total_mask_h
                first_c = False
            else:
                total_mask_c = torch.cat((total_mask_c, total_mask_h), dim = 0)
            first_h = True
        return total_mask_c