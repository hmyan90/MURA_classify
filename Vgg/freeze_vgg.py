from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense, Flatten
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from scipy.misc import imresize
from pathlib import Path

train=pd.read_csv("MURA-v1.1/train_image_paths.csv", names=['filename'])
valid=pd.read_csv("MURA-v1.1/valid_image_paths.csv", names=['filename'])
train_path="./"
valid_path="./"

if Path('Vgg/bottleneck_features_trainX.npy').exists():
    X_train = np.load(open('Vgg/bottleneck_features_trainX.npy'))
    X_valid = np.load(open('Vgg/bottleneck_features_validationX.npy'))
    Y_train = np.load(open('Vgg/bottleneck_features_trainY.npy'))
    Y_valid = np.load(open('Vgg/bottleneck_features_validationY.npy'))    
else:
    train_img=[]
    train_y = []
    for i in range(len(train)):
        temp_img=image.load_img(train_path+train['filename'][i], target_size=(224,224))
        temp_img=image.img_to_array(temp_img)
        train_img.append(temp_img)
        train_y.append(0 if 'negative' in train['filename'][i] else 1)
    
    #converting train images to array and applying mean subtraction processing
    train_img=preprocess_input(np.array(train_img) )
    
    # applying the same procedure with the valid dataset
    valid_img=[]
    valid_y = []
    for i in range(len(valid)):
        temp_img=image.load_img(valid_path+valid['filename'][i],target_size=(224,224))
        temp_img=image.img_to_array(temp_img)
        valid_img.append(temp_img)
        valid_y.append(0 if 'negative' in valid['filename'][i] else 1)
    
    valid_img=preprocess_input(np.array(valid_img) )
    
    # loading VGG16 model weights
    model = VGG16(weights='imagenet', include_top=False)
    # Extracting features from the train dataset using the VGG16 pre-trained model
    
    features_train=model.predict(train_img)
    # Extracting features from the train dataset using the VGG16 pre-trained model
    
    features_valid=model.predict(valid_img)
    
    train_x=features_train.reshape((len(train), 7*7*512))
    train_y=np.asarray(train_y)
    
    valid_x=features_valid.reshape((len(valid), 7*7*512))    
    valid_y=np.asarray(valid_y)
    
    X_train, X_valid, Y_train, Y_valid = train_x, valid_x, train_y, valid_y
    
    np.save(open('Vgg/bottleneck_features_trainX.npy', 'w'), X_train)
    np.save(open('Vgg/bottleneck_features_validationX.npy', 'w'), X_valid)
    np.save(open('Vgg/bottleneck_features_trainY.npy', 'w'), Y_train)
    np.save(open('Vgg/bottleneck_features_validationY.npy', 'w'), Y_valid)

# creating a mlp model
from keras.layers import Dense, Activation
model=Sequential()

model.add(Dense(1000, input_dim=7*7*512, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500, input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(100, input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(1, input_dim=100, activation='sigmoid'))

# why relu, sigmoid, sigmoid, sigmoid have much higher accuracy than other ?
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()

# fitting the model 
model.fit(X_train, Y_train, epochs=30, batch_size=128, verbose=1, validation_data=(X_valid,Y_valid))

