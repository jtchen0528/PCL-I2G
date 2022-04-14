partition=test
gpu=7
checkpoint=checkpoints/PCL-I2G-FF256-32frames-Modified-5e-4
which_epoch=5
dset=/scratch2/users/jtchen0528/Datasets/PatchForensics

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-DF \
	--real_im_path $dset/original/$partition \
	--fake_im_path $dset/DF/$partition \
	--train_config $checkpoint/opt.yml

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-F2F \
	--real_im_path $dset/original/$partition \
	--fake_im_path $dset/F2F/$partition \
	--train_config $checkpoint/opt.yml

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-FS \
	--real_im_path $dset/original/$partition \
	--fake_im_path $dset/FS/$partition \
	--train_config $checkpoint/opt.yml

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-NT \
	--real_im_path $dset/original/$partition \
	--fake_im_path $dset/NT/$partition \
	--train_config $checkpoint/opt.yml