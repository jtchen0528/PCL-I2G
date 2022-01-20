### celebahq-pgan raw samples from the gan vs real images ###
### uses --no_serial_batches as samples are not paired ### 
dset=/scratch2/users/jtchen0528/Datasets/PatchForensics
gpu=7

python train.py \
	--gpu_ids $gpu --seed 0 --loadSize 299 --fineSize 299 \
	--name Xception123_cat-FF-DF-s299-b512-lr5e5 --save_epoch_freq 3 \
 	--real_im_path $dset/original \
 	--fake_im_path $dset/DF \
	--which_model_netD xception_block3_cat_extra1_extra2_extra3 \
	--model patch_discriminator_cat \
	--patience 3 --lr_policy constant --max_epochs 3 \
	--batch_size 32 --lr 5e-5 \
	--overwrite_config