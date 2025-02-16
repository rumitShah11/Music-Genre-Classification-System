import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"
AUDIO_PATH="audio.json"

#using mlp:  a feedforward neural network that generates a set of outputs from a set of inputs

# load dataset
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)    #read data

    #convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

# def load_audio_data(audio_path):
#     with open(audio_path,"r") as fp:
#         audio=json.load(fp)
    
#     x=np.array(audio["mfcc"])
#     x=x[...,np.newaxis]
#     return x

if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH) #load dataset
    
    # split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3) #30% of data will be used to test and 70% used to train


    # build network architecture
    model = keras.Sequential([ #sequential model of the network left->right
        #input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        #1st hidden layer 512 neurons
        keras.layers.Dense(512, activation="relu"),

        #2nd hidden layer 512 neurons
        keras.layers.Dense(256, activation="relu"),

        #3rd hidden layer 512 neurons
        keras.layers.Dense(64, activation="relu"),

        #output layer 10 neuron for 10 Genre
        keras.layers.Dense(10, activation="softmax")
    ]) 

    # compile network
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # train network
    model.fit(inputs_train, targets_train, 
              validation_data = (inputs_test, targets_test),
              epochs = 50, # the number of passes of the entire training dataset the machine learning algorithm has completed
              batch_size =32)   #number of samples to be trained per batch

    test_error,test_accuracy=model.evaluate(inputs_test,targets_test,verbose=1)

    print(test_accuracy)