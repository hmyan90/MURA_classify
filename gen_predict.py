import json
import sys
from keras.models import model_from_json
import pandas as pd
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np

def load_model():
    # load json and create model
    print("Loaded model from disk")
    json_file = open('ResNet/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("ResNet/model.h5")
    loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return loaded_model

def load_test_data(cvs_file_name):
    
    test = pd.read_csv(cvs_file_name, names=['filename'])
    test_img=[]
    for i in range(len(test)):
        temp_img=image.load_img(test['filename'][i], target_size=(224,224))
        temp_img=image.img_to_array(temp_img)
        test_img.append(temp_img)
    
    test_img=preprocess_input(np.array(test_img))
    return test_img

if __name__ == "__main__":
    
    # print command line arguments
    cvs_file_name = sys.argv[1]
    output_cvs_file_path = sys.argv[2]
    
    model = load_model()
    X_test = load_test_data(cvs_file_name)
    
    predictions = model.predict(X_test, verbose=1)
    df = pd.read_csv(cvs_file_name, names=['filename'])
    df['predictions'] = predictions
    df.to_csv(output_cvs_file_path, header=False)


