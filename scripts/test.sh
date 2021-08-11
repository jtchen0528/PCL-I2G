partition=test
gpu=5
checkpoint=checkpoints/PCL-I2G-FF128-5e-5
which_epoch=bestval
#dset=/scratch2/users/jtchen0528/patch-forensics-test-dataset/faces
# dset=/scratch2/users/clairelai/patch-forensics-rp-dataset/faces
# dset=/scratch3/users/clairelai/OpenMFC20_Eval_GAN_Image_Full/probe
# dset=~/Documents/AIIU/Datasets/OpenMFC20_Eval_GAN_Image_Full/probe
# dset=~/Documents/AIIU/Datasets/patch-forensics-rp-dataset/faces
# dset=~/Documents/AIIU/Datasets/patch-forensics-rp-dataset/faceforensics_aligned
dset=/scratch3/users/clairelai/faceforensics_aligned
# dset=~/Documents/AIIU/Datasets/CNNDetection/test_pc

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name openmfc \
# 	--fake_im_path $dset/$partition \
# 	--train_config $checkpoint/opt.yml \

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-DF \
	--real_im_path $dset/Deepfakes/original/$partition \
	--fake_im_path $dset/Deepfakes/manipulated/$partition \
	--train_config $checkpoint/opt.yml

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name FF-FF \
# 	--real_im_path $dset/Face2Face/original/$partition \
# 	--fake_im_path $dset/Face2Face/manipulated/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name FF-FS \
# 	--real_im_path $dset/FaceSwap/original/$partition \
# 	--fake_im_path $dset/FaceSwap/manipulated/$partition \
# 	--train_config $checkpoint/opt.yml 

python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
	--average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FF-NT \
	--real_im_path $dset/NeuralTextures/original/$partition \
	--fake_im_path $dset/NeuralTextures/manipulated/$partition \
	--train_config $checkpoint/opt.yml

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name celebahq-pgan-pretrained \
# 	--real_im_path $dset/celebahq/real-tfr-1024-resized128/$partition \
# 	--fake_im_path $dset/celebahq/pgan-pretrained-128-png/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name celebahq-sgan-pretrained \
# 	--real_im_path $dset/celebahq/real-tfr-1024-resized128/$partition \
# 	--fake_im_path $dset/celebahq/sgan-pretrained-128-png/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name celebahq-glow-pretrained \
# 	--real_im_path $dset/celebahq/real-tfr-1024-resized128/$partition \
# 	--fake_im_path $dset/celebahq/glow-pretrained-128-png/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name celeba-gmm-pretrained \
# 	--real_im_path $dset/celeba/mfa-real/$partition \
# 	--fake_im_path $dset/celeba/mfa-defaults/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name ffhq-pgan-pretrained \
# 	--real_im_path $dset/ffhq/real-tfr-1024-resized-128/$partition \
# 	--fake_im_path $dset/ffhq/pgan-9k-128-png/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name ffhq-sgan-pretrained \
# 	--real_im_path $dset/ffhq/real-tfr-1024-resized-128/$partition \
# 	--fake_im_path $dset/ffhq/sgan-pretrained-128-png/$partition \
# 	--train_config $checkpoint/opt.yml 

# python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name ffhq-sgan2-pretrained \
# 	--real_im_path $dset/ffhq/real-tfr-1024-resized-128/$partition \
# 	--fake_im_path $dset/ffhq/sgan2-pretrained-128-png/$partition \
# 	--train_config $checkpoint/opt.yml 

	# --visualize \

# for set in biggan cyclegan gaugan progan seeingdark stylegan whichfaceisreal crn deepfake imle san stargan stylegan2
# do 
# 	python test.py --which_epoch $which_epoch --gpu_ids $gpu --partition $partition \
# 	--average_mode after_softmax --topn 100 --force_redo \
# 	--dataset_name CNNDetection-$set \
# 	--real_im_path $dset/$set/0_real \
# 	--fake_im_path $dset/$set/1_fake \
# 	--train_config $checkpoint/opt.yml 
# done