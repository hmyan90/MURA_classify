from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
# from sklearn.metrics import log_loss
from scipy.misc import imresize

train=pd.read_csv("MURA-v1.1/train_image_paths.csv", names=['filename'])
valid=pd.read_csv("MURA-v1.1/valid_image_paths.csv", names=['filename'])
train_path="./"
valid_path="./"

train_img=[]
train_y = []
for i in range(len(train)):
    temp_img=image.load_img(train_path+train['filename'][i], target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)
    train_y.append(0 if 'negative' in train['filename'][i] else 1)

# converting train images to array and applying mean subtraction processing
train_img=preprocess_input(np.array(train_img) )

# applying the same procedure with the valid dataset
valid_img=[]
valid_y = []
for i in range(len(valid)):
    temp_img=image.load_img(valid_path+valid['filename'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    valid_img.append(temp_img)
    valid_y.append(0 if 'negative' in valid['filename'][i] else 1)

valid_img=preprocess_input(np.array(valid_img))

X_train, X_valid, Y_train, Y_valid = train_img, valid_img, np.asarray(train_y), np.asarray(valid_y)


from keras.models import Model

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    model = VGG16(weights='imagenet', include_top=True)
    
    # pop last layer, use own layer inorder use num_classes
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    x=Dense(num_classes, activation='softmax')(model.output)
    model=Model(model.input, x)

    #To set the first 8 layers to non-trainable (weights will not be updated)
    print("Total layer: %d" %len(model.layers))
    for layer in model.layers[:8]:
        layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 1
batch_size = 64
nb_epoch = 10

# Load our model
model = vgg16_model(img_rows, img_cols, channel, num_classes)

model.summary()

# Start Fine-tuning
model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, Y_valid))
