partition=test
gpu=7
checkpoint=checkpoints/resnet123_cat-FF-DF-s256-b64-lr5e5
which_epoch=bestval
dset=/scratch2/users/jtchen0528/Datasets/PatchForensics

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name FF-DF \
# 	--real_im_path $dset/original/$partition \
# 	--fake_im_path $dset/DF/$partition \
# 	--train_config $checkpoint/opt.yml

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-F2F \
	--real_im_path $dset/original/$partition \
	--fake_im_path $dset/F2F/$partition \
	--train_config $checkpoint/opt.yml

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name FF-FS \
# 	--real_im_path $dset/original/$partition \
# 	--fake_im_path $dset/FS/$partition \
# 	--train_config $checkpoint/opt.yml

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name FF-NT \
# 	--real_im_path $dset/original/$partition \
# 	--fake_im_path $dset/NT/$partition \
# 	--train_config $checkpoint/opt.yml