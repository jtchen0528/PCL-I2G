# face segmentation model
git clone https://github.com/zllrunning/face-parsing.PyTorch.git face_parsing_pytorch
gdown https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
mkdir -p face_parsing_pytorch/res/cp
mv 79999_iter.pth face_parsing_pytorch/res/cp

# dlib facial landmarks predictor
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2