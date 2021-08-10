import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import sys
from utils import pidfile
import numpy as np
from data.processing.celebahq_crop import celebahq_crop, I2G_crop
import pickle
import cv2

parser = argparse.ArgumentParser(description='Process and align face forensics frames')
parser.add_argument('--source_dir_original', required=True, help='source videos directory, e.g. original_sequences/youtube/c23/videos')
parser.add_argument('--outsize', type=int, default=128, help='resize to this size')
parser.add_argument('--output_dir', required=True, help='output directory')
parser.add_argument('--split', default='val.json', help='Path to split json file')

args = parser.parse_args()
rnd = np.random.RandomState(0)

with open(args.split) as f:
    split = json.load(f)
split_name = os.path.splitext(os.path.basename(args.split))[0]

outdir = args.output_dir
os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join(outdir, 'original', split_name), exist_ok=True)
os.makedirs(os.path.join(outdir, 'detections', split_name), exist_ok=True)
mask_landmark_list = {}

for i, s in enumerate(tqdm(split)):
    vidname = '_'.join(s)
    vidname_orig = s[0] # take target sequence for original videos
    print("%d: %s" % (i, vidname))
    vidpath_orig = os.path.join(args.source_dir_original, vidname_orig)

    original_video_frames = os.listdir(vidpath_orig)

    counter = 0
    for j, orig_frame in enumerate(original_video_frames):
        orig_frame_path = os.path.join(args.source_dir_original, vidname_orig, orig_frame)
        print(orig_frame_path)
        orig = cv2.imread(orig_frame_path, cv2.IMREAD_UNCHANGED)
        if os.path.isfile(os.path.join(outdir, 'detections', split_name,
                                       '%s_%03d.npz' % (vidname, j))):
            print('Found existing %s_%03d.npz' % (vidname, j))
            counter += 1
            continue
        try:
            # might return none or out of bounds error
            # use original landmarks
            cropped_orig, landmarks = celebahq_crop(orig)
            landmarks = I2G_crop(orig)
            cropped_orig = cropped_orig.resize((args.outsize, args.outsize), Image.LANCZOS)

            cropped_orig.save(os.path.join(outdir, 'original', split_name,
                                           '%s_%03d.png' % (vidname, j)))
            np.savez(os.path.join(outdir, 'detections', split_name,
                                  '%s_%03d.npz' % (vidname, j)),
                     lm=landmarks)
            counter += 1
            mask_landmark_list['%s_%03d.png' % (vidname, j)] = landmarks

            # for val/test partitions, just take 100 detected frames per video
            if counter == 100 and split_name in ['test', 'val']:
                print("Processed 100 frames of %s" % vidname)
                print("Moving to next video")
                break
        except:
            print("Error:", sys.exc_info()[0])
fh = open(os.path.join(outdir, 'original', split_name, split_name + '_landmark.pkl'), 'wb')
pickle.dump(mask_landmark_list, fh)
