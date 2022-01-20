### celebahq-pgan raw samples from the gan vs real images ###
### uses --no_serial_batches as samples are not paired ### 
dset=/scratch2/users/jtchen0528/Datasets/PatchForensics
gpu=7

python train.py \
	--gpu_ids $gpu --seed 0 --loadSize 299 --fineSize 299 \
	--name PCL-I2G-FF128-32frames-Modified-5e-5 --save_epoch_freq 10 \
 	--real_im_path $dset/original \
 	--fake_im_path $dset/DF \
	--which_model_netD xception_block2_extra2 --model patch_discriminator_multihead_selfattention --lbda 10 \
	--patience 5 --lr_policy constant --max_epochs 5 --batch_size 32 --lr 5e-5 \
	--overwrite_config