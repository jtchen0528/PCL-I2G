### Dataset processing ###
manipulated=/scratch2/users/jtchen0528/Datasets/FaceForensics++_redownload/manipulated_sequences/
original=/scratch2/users/jtchen0528/Datasets/FaceForensics++_redownload/original_sequences/youtube/c23/frames
out=/scratch2/users/jtchen0528/Datasets/PatchForensics
outsize=256

python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated $manipulated \
	--source_dir_original $original \
	--outsize $outsize \
	--output_dir $out\
	--split resources/splits/train.json

python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated $manipulated \
	--source_dir_original $original \
	--outsize $outsize \
	--output_dir $out\
	--split resources/splits/val.json

python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated $manipulated \
	--source_dir_original $original \
	--outsize $outsize \
	--output_dir $out\
	--split resources/splits/test.json