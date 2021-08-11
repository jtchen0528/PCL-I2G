### Deepfakes processing ###

# # # train 
python -m data.processing.faceforensics_process_frames_I2G_2 \
	--source_dir_manipulated ~/Documents/AIIU/Datasets/faceforensics/NT \
	--source_dir_original ~/Documents/AIIU/Datasets/faceforensics/original \
	--outsize 256 \
	--output_dir ~/Documents/AIIU/Datasets/faceforensics_aligned/NT \
	--split resources/splits/train.json

python -m data.processing.faceforensics_process_frames_I2G_2 \
	--source_dir_manipulated ~/Documents/AIIU/Datasets/faceforensics/NT \
	--source_dir_original ~/Documents/AIIU/Datasets/faceforensics/original \
	--outsize 256 \
	--output_dir ~/Documents/AIIU/Datasets/faceforensics_aligned/NT \
	--split resources/splits/val.json

python -m data.processing.faceforensics_process_frames_I2G_2 \
	--source_dir_manipulated ~/Documents/AIIU/Datasets/faceforensics/NT \
	--source_dir_original ~/Documents/AIIU/Datasets/faceforensics/original \
	--outsize 256 \
	--output_dir ~/Documents/AIIU/Datasets/faceforensics_aligned/NT \
	--split resources/splits/test.json