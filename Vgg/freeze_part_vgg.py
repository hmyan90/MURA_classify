from datetime import datetime
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.utils import class_weight
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# train=pd.read_csv("MURA-v1.1/train_image_paths.csv", names=['filename'])
# valid=pd.read_csv("MURA-v1.1/valid_image_paths.csv", names=['filename'])
# train_path="./"
# valid_path="./"

# train_img=[]
# train_y = []
# for i in range(len(train)):
#     temp_img=image.load_img(train_path+train['filename'][i], target_size=(224,224))
#     temp_img=image.img_to_array(temp_img)
#     train_img.append(temp_img)
#     train_y.append(0 if 'negative' in train['filename'][i] else 1)
# 
# # converting train images to array and applying mean subtraction processing
# train_img=preprocess_input(np.array(train_img) )
# 
# # applying the same procedure with the valid dataset
# valid_img=[]
# valid_y = []
# for i in range(len(valid)):
#     temp_img=image.load_img(valid_path+valid['filename'][i],target_size=(224,224))
#     temp_img=image.img_to_array(temp_img)
#     valid_img.append(temp_img)
#     valid_y.append(0 if 'negative' in valid['filename'][i] else 1)
# 
# valid_img=preprocess_input(np.array(valid_img))
# 
# X_train, X_valid, Y_train, Y_valid = train_img, valid_img, np.asarray(train_y), np.asarray(valid_y)

img_size = 224
batch_size = 32
channel = 3
num_classes = 1

# We then scale the variable-sized images to 224x224
# We augment .. by applying random lateral inversions and rotations.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    shuffle=True,
    target_size=(img_size, img_size),
    class_mode='binary',
    batch_size=batch_size,)

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    'data/val',
    shuffle=True,  # otherwise we get distorted batch-wise metrics
    class_mode='binary',
    target_size=(img_size, img_size),
    batch_size=batch_size,)

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    base_resNet = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channel))
    
    x = base_vggNet.output # (1, 1, 2048)
    
#     x = Flatten()(x) # 2048
    x = GlobalAveragePooling2D()(x) # 2048
    
    # add fully-connected & dropout layers
#     x = Dense(500, activation='relu',name='fc-1')(x)
#     x = Dropout(0.5)(x)
    
    # add another fully-connected & dropout layers
#     x = Dense(100, activation='sigmoid',name='fc-2')(x)
#     x = Dropout(0.5)(x)
    
    # a softmax layer for 1 class
    x = Dense(num_classes, activation='sigmoid',name='output_layer')(x) # softmax???
    
    # TODO , try SVM
    
    transfer_vggNet = Model(inputs=base_vggNet.input, outputs=x)
    
#     for layer in transfer_resNet.layers[:-4]:
#         layer.trainable = False
            
    transfer_vggNet.summary()
    
#     sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=1e-4, decay=1e-4, beta_1=0.9, beta_2=0.999)
    transfer_vggNet.compile(optimizer=adam, loss=binary_crossentropy, metrics=[binary_accuracy])

    return transfer_vggNet


nb_epoch = 200

model = vgg16_model(img_size, img_size, channel, num_classes)

checkpoint = ModelCheckpoint(filepath='Vgg/model2_weight.hdf5', verbose=1, save_best_only=True)
early_stop = EarlyStopping(patience=10)
now_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
tensorboard = TensorBoard(log_dir='Vgg/logs/{}/'.format(now_iso))
callbacks = [checkpoint, tensorboard, checkpoint]
    
print("Saved model to disk")
model_json = model.to_json()
with open("Vgg/model2.json", "w") as json_file:
    json_file.write(model_json)
    
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=nb_epoch,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
#     class_weight=weights,
    workers=4,
    use_multiprocessing=True,
    verbose=1,
    callbacks=callbacks)

# model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
