### celebahq-pgan raw samples from the gan vs real images ###
### uses --no_serial_batches as samples are not paired ### 
dset=/scratch2/users/jtchen0528/Datasets/PatchForensics
gpu=6

python train.py \
	--gpu_ids $gpu --seed 0 --loadSize 299 --fineSize 299 \
	--name Xception12_MSA-FF-DF-s299-b64-lr5e5 --save_epoch_freq 10 \
 	--real_im_path $dset/original \
 	--fake_im_path $dset/DF \
	--which_model_netD xception_block2_extra2 --model patch_discriminator_multihead_selfattention --lbda 10 \
	--patience 5 --lr_policy constant --max_epochs 200 --batch_size 64 --lr 5e-5 \
	--overwrite_config

python train.py \
	--gpu_ids $gpu --seed 0 --loadSize 299 --fineSize 299 \
	--name Xception12_MSA-FF-F2F-s299-b64-lr5e5 --save_epoch_freq 10 \
 	--real_im_path $dset/original \
 	--fake_im_path $dset/F2F \
	--which_model_netD xception_block2_extra2 --model patch_discriminator_multihead_selfattention --lbda 10 \
	--patience 5 --lr_policy constant --max_epochs 200 --batch_size 64 --lr 5e-5 \
	--overwrite_config

python train.py \
	--gpu_ids $gpu --seed 0 --loadSize 299 --fineSize 299 \
	--name Xception12_MSA-FF-FS-s299-b64-lr5e5 --save_epoch_freq 10 \
 	--real_im_path $dset/original \
 	--fake_im_path $dset/FS \
	--which_model_netD xception_block2_extra2 --model patch_discriminator_multihead_selfattention --lbda 10 \
	--patience 5 --lr_policy constant --max_epochs 200 --batch_size 64 --lr 5e-5 \
	--overwrite_config

python train.py \
	--gpu_ids $gpu --seed 0 --loadSize 299 --fineSize 299 \
	--name Xception12_MSA-FF-NT-s299-b64-lr5e5 --save_epoch_freq 10 \
 	--real_im_path $dset/original \
 	--fake_im_path $dset/NT \
	--which_model_netD xception_block2_extra2 --model patch_discriminator_multihead_selfattention --lbda 10 \
	--patience 5 --lr_policy constant --max_epochs 200 --batch_size 64 --lr 5e-5 \
	--overwrite_config