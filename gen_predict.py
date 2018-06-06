# import json
import sys
import keras
from keras.models import model_from_json
import pandas as pd
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.metrics import binary_accuracy, binary_crossentropy


def load_model():
    # load json and create model
    print("Loaded model from disk")
    json_file = open('ResNet/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("ResNet/model2_weight.hdf5")
    adam = keras.optimizers.Adam(lr=1e-4, decay=1e-4, beta_1=0.9, beta_2=0.999)
    loaded_model.compile(optimizer=adam, loss=binary_crossentropy, metrics=[binary_accuracy])

    return loaded_model

def load_test_data(cvs_file_name):
    
    test = pd.read_csv(cvs_file_name, names=['img_name'])
    test_img=[]
    for i in range(len(test)):
        temp_img=image.load_img(test['img_name'][i], target_size=(224,224))
        temp_img=image.img_to_array(temp_img)
        test_img.append(temp_img)
    
    test_img=preprocess_input(np.array(test_img))
    return test_img

def _study_name(img_name):
    
    idx = img_name.rfind("/")
    return img_name[:idx+1]
    
def mergeInStudy(df):
    
    result_studies = [] 
    result_predicts = []
    
    count = 0
    tmp = []
    for i in range(0, df.img_name.size):
        tmp.append(df.predictions[i])
        if i == df.img_name.size-1 or _study_name(df.img_name[i]) != _study_name(df.img_name[i+1]):
            # merge and get result
            print tmp
            result_studies.append(_study_name(df.img_name[i]))
            pred = 1 if sum(tmp)/len(tmp) > 0.5 else 0
            result_predicts.append(pred)
            tmp[:]=[]
            
    return pd.DataFrame({"study": result_studies, "predictions": result_predicts})

    
if __name__ == "__main__":
    
    # print command line arguments
    cvs_file_name = sys.argv[1]
    output_cvs_file_path = sys.argv[2]
    
    model = load_model()
    X_test = load_test_data(cvs_file_name)
    
    predictions = model.predict(X_test, verbose=1)
#     print predictions.shape
    df = pd.read_csv(cvs_file_name, names=['img_name'])
    df['predictions'] = predictions
#     print df
    df = mergeInStudy(df)
    print df
    df.to_csv(output_cvs_file_path, header=False, columns=["study", "predictions"])
