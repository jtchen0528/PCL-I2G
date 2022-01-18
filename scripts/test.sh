partition=test
gpu=5
checkpoint=checkpoints/PCL-I2G-FF128-32frames-Modified-5e-5
which_epoch=bestval
dset=/scratch2/users/jtchen0528/Datasets/PatchForensics

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-DF \
	--real_im_path $dset/original/$partition \
	--fake_im_path $dset/DF/$partition \
	--train_config $checkpoint/opt.yml
