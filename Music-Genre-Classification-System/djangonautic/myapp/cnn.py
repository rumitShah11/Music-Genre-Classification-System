import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import os

DATA_PATH=os.path.dirname(os.path.realpath(__file__))+"/data.json"
AUDIO_PATH=os.path.dirname(os.path.realpath(__file__))+"/audio.json"

def load_data(data_path):

    with open(data_path,"r") as fp:
        data=json.load(fp)

    x=np.array(data["mfcc"])
    y=np.array(data["labels"])

    return x,y

def load_audio_data(audio_path):
    with open(audio_path,"r") as fp:
        audio=json.load(fp)
    
    x=np.array(audio["mfcc"])
    x=x[...,np.newaxis]
    return x

def prepare_datasets(test_size,validation_size):

    #load data
    X,y=load_data(DATA_PATH)

    #create train/test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)

    #create train/validation split
    X_train,X_validation,y_train,y_validation=train_test_split(X_train,y_train,test_size=validation_size)

    #tf require 3-d array for CNN
    X_train=X_train[...,np.newaxis]
    X_validation=X_validation[...,np.newaxis]
    X_test=X_test[...,np.newaxis]

    return X_train,X_validation,X_test,y_train,y_validation,y_test

def build_model(input_shape):
    #build model
    model=keras.Sequential()

    #1st layer (add layer with 32 kernels of grid size 3x3)
    model.add(keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same")) #Pooling layers are used to reduce the dimensions, Stride are number of pixels shifts over the input matrix
    model.add(keras.layers.BatchNormalization())

    #2nd layer
    model.add(keras.layers.Conv2D(32,(3,3),activation="relu"))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same")) 
    model.add(keras.layers.BatchNormalization())

    #3rd layer
    model.add(keras.layers.Conv2D(32,(2,2),activation="relu"))
    model.add(keras.layers.MaxPool2D((2,2),strides=(2,2),padding="same"))
    model.add(keras.layers.BatchNormalization())

    #flatten the output
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation="relu")) #layer for classification 64 neurons
    model.add(keras.layers.Dropout(0.3)) 

    #output layer
    model.add(keras.layers.Dense(10,activation="softmax")) #10 neurons for 10 different genres
    #softmax makes a probability distribution for each genre, genre with highest score is the answer

    return model

def predict(model,X):

    X=X[np.newaxis,...]
    
    prediction=model.predict(X)


    #get index with max value
    predictIndex=np.argmax(prediction, axis=1)

    print(predictIndex)
    return predictIndex


def result(index):
    with open(DATA_PATH,"r") as fp:
        data=json.load(fp)
    print(data['mapping'][index[0]])
    return data["mapping"][index[0]]
    
    

def classify():

    #create train,test,validation data set
    #validation will be used to evaluate the model

    X_train,X_validation,X_test,y_train,y_validation,y_test=prepare_datasets(0.25,0.2) #25% test size (how much training set will b used for test) 20% validation size (20% training set used for validation)

    input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])

    #build the CNN network
    model=build_model(input_shape)
    # model = keras.models.load_model(os.path.dirname(os.path.realpath(__file__)))

    #compile the model
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    #train the model
    model.fit(X_train,y_train,validation_data=(X_validation,y_validation),batch_size=32,epochs=40)

    #evluate CNN on test set
    test_error,test_accuracy=model.evaluate(X_test,y_test,verbose=1)
    print("Accuracy on test set is:{}".format(test_accuracy))

    #make predictions
    test=load_audio_data(AUDIO_PATH)
    X= X_test[100]
    y=y_test[100]
    # index=predict(model,X,y)
    index=predict(model,test[0])
    return result(index)
    

    
    

    