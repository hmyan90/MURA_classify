# MURA_classify

### Your directory looks like:
               ├── MURA-v1.1/
               
MURA_classify  ├── ResNet/

               ├── Vgg/
               
               └── gen_predict.py
             
                 
           
### Run
  cd MURA_classify/
  * python prehandle_data.py (do first, do once)
  
  * python ResNet/freeze_resnet.py
  
  * python ResNet/freeze_part_resnet.py
  
  * python vgg/freeze_vgg.py
  
  * python vgg/freeze_part_vgg.py
  
  
### Prediction

  * python ResNet/gen_predict.py MURA-v1.1/valid_image_paths.csv ./prediction.csv
