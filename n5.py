# NOTES:
# using ham + spam files in a 80 to 20 training to testing ratio

#necessary import statements to run this program
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import email
import email.policy
from bs4 import BeautifulSoup
import logging
import clean

logging.disable()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# CONSTANT VARIABLES

NUM_EPOCHS = 5
BATCHSIZE = 200
FILENAME = "file_busters_model.h5"

# # used in preprocess() -- returns file and label
# # labels: 0 = ham -- 1 = spam
# def labeler(example, index):
#         return example, tf.cast(index, tf.int64)  

# ------------- PREPROCESS/CLEAN DATA -------------------

def preprocess():
    ham_location = "C:\\Users\\Student\\Desktop\\email_files\\ham"
    spam_location = "C:\\Users\\Student\\Desktop\\email_files\\spam"

    numTrainHam = round(500*0.8)
    numTrainSpam = round(2550*0.8)
    train_emails = []
    test_emails = []
    trainHam = 0
    trainSpam = 0
    testHam = 0
    testSpam = 0

    # open ham email text file and add to ham train or test list
    for x in range(1, 2551):
        filename = str(x) + '.txt'
        text_dir = ham_location + '\\' + filename
        content = open(text_dir, encoding="utf8")
        content = content.read()

        if x < numTrainHam:        
            train_emails.append(content)
            trainHam+=1    
        else:
            test_emails.append(content)
            testHam+=1

    # open spam email text file and add to spam test or train list
    for i in range(1, 501):
        filename = str(i) + '.txt'
        text_dir = spam_location + '\\' + filename
        content = open(text_dir, encoding="utf8")
        content = content.read()

        if i < numTrainSpam:        
            train_emails.append(content)
            trainSpam+=1
        else:
            test_emails.append(content)
            testSpam+=1

    train_email_labels = []
    test_email_labels = []
    #train_labels 
    for x in range(0, trainHam):
        train_email_labels.append("ham")
        

    for x in range(0, trainSpam):
        train_email_labels.append("spam") 

    #test_labels 
    for x in range(0, testHam):
        test_email_labels.append("ham")

    for x in range(0, testSpam):
        test_email_labels.append("spam")   


    # ----- CREATE TRAIN DATAFRAME --------

    # creates pandas dataframe with 2 columns: email and labels
    emails_df = pd.DataFrame(train_emails, columns=['emails'])
    emails_df['labels'] = train_email_labels  

    # pops out values of labels column
    labels = emails_df.pop('labels')
    # creates the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((emails_df.values, labels.values))

    # ----- CREATE TEST DATAFRAME ---------

    # creates pandas dataframe with 2 columns: email and labels
    test_emails_df = pd.DataFrame(test_emails, columns=['emails'])
    test_emails_df['labels'] = test_email_labels 

    # pops out values of labels column
    test_labels = test_emails_df.pop('labels')
    # creates the training dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_emails_df.values, test_labels.values))

    return train_dataset, test_dataset


# ------------ TRAIN -----------------------

def train(train_data):
    # try to load already saved model
    try:
        # this line will throw error if file doesn't exist
        model=tf.keras.models.load_model(FILENAME)
    except:

        print('a')
        model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        print('b')
        
        hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], 
                                dtype=tf.string, trainable=True)
        print('c')   

        #create model
        model = keras.Sequential([ # Sequential means sequence of layers
            hub_layer,
            # learn about the datapoints in relationship to the datapoints that came before it and after it
            # keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            # 128 neurons, rectified linear unit
            keras.layers.Dense(128, activation="relu"),        

            # num of output classes, softmax probability dist (softmax = softens max values)
            keras.layers.Dense(2, activation="softmax")
            ])
        print('d')

        #compile model
        # model.compile(optimizer="adam", loss="binary_crossentropy",
        # metrics=["accuracy"])
        model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
        print('e')

        # fit model
        model.fit(train_data, epochs=NUM_EPOCHS)
        print('f')
        # save model
        model.save(FILENAME)
        print('g')

    return model


# -------------- TEST ---------------------

# checks to see if the model is able to correctly predict the type (spam/ham) of email
# with un-seen data
def predict(model, test_data):
    # get accuracy and metrics from model
    # params: test input and output
    # returns loss and accuracy -> wrong vs right
    test_loss, test_accuracy = model.evaluate(test_data) 
    print("Accuracy", test_accuracy, "\nLoss", test_loss)

    # use test data to predict output
    predictions = model.predict(test_data[:10])
    # for i in range(5):
    #     #print prediction and actual
    #     print("Actual:", test_labels[i], "Expected:", np.argmax(predictions[i]))

# preprocess data
train_data, test_data = preprocess()

# train the data
model = train(train_data)


# test model
predict(model, test_data)

logging.debug('End Program') #signifies the end of the program through the terminal

