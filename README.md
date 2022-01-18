# PCL-I2G
Unofficial implementation of paper: [Learning Self-Consistency for Deepfake Detection](https://arxiv.org/pdf/2012.09311.pdf) (ICCV2021)

## Installation
1. create conda environment with Python=3.7
    ```bash
    conda create -n PCL-I2G python=3.7
    conda activate PCL-I2G
    ```
2. install pytorch 1.9.0, torchvision 0.10.0 with compatible cuda version
    ```bash
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```
3. install required packages
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
Basically any real data works on the methodology, but here I use FaceForensics++.
* [FaceForensics++](https://github.com/ondyari/FaceForensics)

## Pre-processing
1. extract frames from videos (with [ffmpeg](https://www.ffmpeg.org/))
2. run data/preprocessing/faceforensics_process_frames.py
    ```bash
    python -m data.processing.faceforensics_process_frames.py \
        --source_dir_manipulated $manipulated \
        --source_dir_original $original \
        --outsize $outsize \
        --output_dir $out\
        --split resources/splits/train.json
        # --split resources/splits/val.json
        # --split resources/splits/test.json
    ```
    or run scripts/preprocess.sh
    ```bash
    bash scripts/train.sh
    ```

## Inconsistency Image Generator (I2G)

1. run generate_I2G.py
    ```bash
    python generate_I2G.py \
        --real_im_path $original \
        --batch_size 512 \
        --out_size 256 \
        --output_dir $out\
        --max_dataset_size 1000
    ```

## Pair-Wise Self-Consistency Learning (PCL)

### Training
    run train_I2G.py with specific setting: 
    ```bash
        --which_model_net resnet34_layer4_extra3 \ 
        --model patch_inconsistency_discriminator \ 
        --lbda 10
    ```
    or run
    ```bash
    python train_I2G.py \
        --gpu_ids $gpu --seed 0 --loadSize 256 --fineSize 256 \
        --name PCL-I2G-FF256-32frames-5e-5 --save_epoch_freq 10 \
        --real_im_path $dset \
        --which_model_netD resnet34_layer4_extra3 --model patch_inconsistency_discriminator --lbda 10 \
        --patience 5 --lr_policy constant --max_epochs 1000 --batch_size 512 --lr 5e-5 \
        --overwrite_config
    ```
    or run bash train.sh
    ```bash
    bash scrips/train.sh
    ```

### Testing
    run test.py
    ```bash
    python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
        --average_mode after_softmax --topn 100 --force_redo \
        --dataset_name FF-DF \
        --real_im_path $dset/original/$partition \
        --fake_im_path $dset/DF/$partition \
        --train_config $checkpoint/opt.yml
    ```