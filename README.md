# MURA_classify

# Your directory looks like:

(MURA_classify) \ 
  MURA-v1.1/
  ResNet/
  vgg/
  
Run:
  cd MURA_classify/
  python ResNet/freeze_resnet.py
  python ResNet/freeze_part_resnet.py
  python vgg/freeze_vgg.py
  python vgg/freeze_part_vgg.py
  
Prediction:
  python ResNet/gen_predict.py MURA-v1.1/valid_image_paths.csv prediction.csv
