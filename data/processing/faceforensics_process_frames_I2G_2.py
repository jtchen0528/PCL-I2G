import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import skvideo.io
import json
import sys
from utils import pidfile
import numpy as np
from data.processing.celebahq_crop import celebahq_crop
from skimage import io

parser = argparse.ArgumentParser(description='Process and align face forensics frames')
parser.add_argument('--source_dir_manipulated', required=True, help='source videos directory, e.g. manipulated_sequences/Deepfakes/c23/videos')
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
os.makedirs(os.path.join(outdir, 'DF', split_name), exist_ok=True)
os.makedirs(os.path.join(outdir, 'F2F', split_name), exist_ok=True)
os.makedirs(os.path.join(outdir, 'FS', split_name), exist_ok=True)
os.makedirs(os.path.join(outdir, 'NT', split_name), exist_ok=True)
# os.makedirs(os.path.join(outdir, 'detections', split_name), exist_ok=True)

for i, s in enumerate(tqdm(split)):
    vidname = '_'.join(s)
    vidname_orig = s[0] # take target sequence for original videos
    # print("%d: %s" % (i, vidname))
    vidpath = os.path.join(args.source_dir_manipulated, 'DF', vidname)
    vidpath_orig = os.path.join(args.source_dir_original, vidname_orig)

    video_frames = os.listdir(vidpath)
    original_video_frames = os.listdir(vidpath_orig)

    counter = 0
    for j, (frame, orig) in enumerate(zip(video_frames, original_video_frames)):
        if os.path.isfile(os.path.join(outdir, 'original', split_name,
                                       '%s_%03d.png' % (vidname, j))):
            print('Found existing %s_%03d.png' % (vidname, j))
            counter += 1
            continue
        try:
            frame_path_DF = os.path.join(args.source_dir_manipulated, 'DF', vidname, frame)
            frame_path_F2F = os.path.join(args.source_dir_manipulated, 'F2F', vidname, frame)
            frame_path_FS = os.path.join(args.source_dir_manipulated, 'FS', vidname, frame)
            frame_path_NT = os.path.join(args.source_dir_manipulated, 'NT', vidname, frame)
            orig_frame_path = os.path.join(args.source_dir_original, vidname_orig, orig)
            frame_DF = io.imread(frame_path_DF)
            frame_F2F = io.imread(frame_path_F2F)
            frame_FS = io.imread(frame_path_FS)
            frame_NT = io.imread(frame_path_NT)
            orig = io.imread(orig_frame_path)
            # might return none or out of bounds error
                # use original landmarks
            cropped_orig, landmarks = celebahq_crop(orig)
            cropped_orig = cropped_orig.resize((args.outsize, args.outsize), Image.LANCZOS)
            cropped_DF = (celebahq_crop(frame_DF, landmarks)[0]
                        .resize((args.outsize, args.outsize), Image.LANCZOS))
            cropped_F2F = (celebahq_crop(frame_F2F, landmarks)[0]
                        .resize((args.outsize, args.outsize), Image.LANCZOS))
            cropped_FS = (celebahq_crop(frame_FS, landmarks)[0]
                        .resize((args.outsize, args.outsize), Image.LANCZOS))
            cropped_NT = (celebahq_crop(frame_NT, landmarks)[0]
                        .resize((args.outsize, args.outsize), Image.LANCZOS))
            # save the results
            cropped_DF.save(os.path.join(outdir, 'DF', split_name,
                                      '%s_%03d.png' % (vidname, j)))
            cropped_F2F.save(os.path.join(outdir, 'F2F', split_name,
                                      '%s_%03d.png' % (vidname, j)))
            cropped_FS.save(os.path.join(outdir, 'FS', split_name,
                                      '%s_%03d.png' % (vidname, j)))
            cropped_NT.save(os.path.join(outdir, 'NT', split_name,
                                      '%s_%03d.png' % (vidname, j)))
            cropped_orig.save(os.path.join(outdir, 'original', split_name,
                                           '%s_%03d.png' % (vidname, j)))
            # np.savez(os.path.join(outdir, 'detections', split_name,
            #                       '%s_%03d.npz' % (vidname, j)),
            #          lm=landmarks)
            counter += 1

            # for val/test partitions, just take 100 detected frames per video
            if counter == 100 and split_name in ['test', 'val']:
                print("Processed 100 frames of %s" % vidname)
                print("Moving to next video")
                break
        except:
            print("Error:", sys.exc_info()[0])
