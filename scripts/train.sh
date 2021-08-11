### celebahq-pgan raw samples from the gan vs real images ###
### uses --no_serial_batches as samples are not paired ### 
dset=/scratch3/users/clairelai/faceforensics_aligned
# dset=/scratch2/users/clairelai/patch-forensics-rp-dataset/faces
# dset=datasets-sample/celebahq
# dset=~/Documents/AIIU/Datasets/CNNDetection
# dset=~/Documents/AIIU/Datasets/FaceForensics++
gpu=8

python train_I2G.py \
	--gpu_ids $gpu --seed 0 --loadSize 128 --fineSize 128 \
	--name PCL-I2G-FF128-32frames-5e-5 --save_epoch_freq 10 \
 	--real_im_path $dset/Deepfakes/original \
 	--fake_im_path $dset/Deepfakes/manipulated \
	--which_model_netD resnet34_layer4_extra3 --model patch_inconsistency_discriminator --lbda 10 \
	--patience 5 --lr_policy constant --max_epochs 1000 --batch_size 512 --lr 5e-5 \
	--overwrite_config