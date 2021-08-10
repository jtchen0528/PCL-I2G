### Deepfakes processing ###

# # train 
# python -m data.processing.faceforensics_process_frames_I2G \
# 	--source_dir_original ~/Documents/AIIU/Datasets/faceforensics/original \
# 	--outsize 256 --split ~/Documents/AIIU/ForgeryTasks/Pair-Wise-Self-Consistency-Learning-Inconsistency-Image-Generator/resources/splits/train.json \
# 	--output_dir ~/Documents/AIIU/Datasets/faceforensics/original_aligned
	
# # val
python -m data.processing.faceforensics_process_frames_I2G \
	--source_dir_original ~/Documents/AIIU/Datasets/faceforensics/original \
	--outsize 256 --split ~/Documents/AIIU/ForgeryTasks/Pair-Wise-Self-Consistency-Learning-Inconsistency-Image-Generator/resources/splits/val.json \
	--output_dir ~/Documents/AIIU/Datasets/faceforensics/original_aligned

# # # test
# python -m data.processing.faceforensics_process_frames_I2G \
# 	--source_dir_original ~/Documents/AIIU/Datasets/faceforensics/original \
# 	--outsize 256 --split ~/Documents/AIIU/ForgeryTasks/Pair-Wise-Self-Consistency-Learning-Inconsistency-Image-Generator/resources/splits/test.json \
# 	--output_dir ~/Documents/AIIU/Datasets/faceforensics/original_aligned
	