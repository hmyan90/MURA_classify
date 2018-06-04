# MURA_classify

Your code and data directory looks like:

  MURA-v1.1/
  ResNet/
  vgg/
  
Run:
  python ResNet/freeze_resnet.py
  python ResNet/freeze_part_resnet.py
  python vgg/freeze_vgg.py
  python vgg/freeze_part_vgg.py
  
Prediction:
  python ResNet/gen_predict.py MURA-v1.1/valid_image_paths.csv prediction.csv
