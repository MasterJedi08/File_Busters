# NOTES:
# hamnspam files as training
# ham / spam files as test


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import email
import email.policy
from bs4 import BeautifulSoup

#split into test and train, input (images) and output (labels)
# CONSTANTS
NUM_EPOCHS = 5
FILENAME = "nn_model.h5"

def preprocess(data):
    # get labels and images for testing and training
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    # normalize data -- don't want super large numbers
    # normalize data by scaling all grayscale values down by 255.0
    train_images = train_images/255.0
    test_images = test_images/255.0
    return (train_images, train_labels, test_images, test_labels)

def train(train_images, train_labels):
    # try to load already saved model
    try:
        # this line will throw error if file doesn't exist
        model=tf.keras.models.load_model(FILENAME)
    except:
        #create model
        model = keras.Sequential([ # Sequential means sequence of layers
            # convery input from 2D list to 1D list
            keras.layers.Flatten(input_shape=(28, 28)),
            # 128 neurons, rectified linear unit
            keras.layers.Dense(activation="relu"), 
            #  outpu classes, softmax probability dist (softmax = softens max values)
            keras.layers.Dense(10, activation="softmax")
            ])

        #compile model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

        # fit model
        model.fit(train_images, train_labels, epochs=NUM_EPOCHS)

        # save model
        model.save(FILENAME)

    return model

def predict(model, test_images, test_labels):
    # get accuracy and metrics from model
    # params: test input and output
    # returns loss and accuracy -> wrong vs right
    test_loss, test_accuracy = model.evaluate(test_images, test_labels) 
    print("Accuracy", test_accuracy)

    # use test data to predict output
    predictions = model.predict(test_images[:10])
    for i in range(5):
        #print prediction and actual
        print("Actual:", test_labels[i], "Expected:", np.argmax(predictions[i]))


# Load dataset
# location test data
os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam')
 
ham_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham')) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam')) if len(name) > 20]


def load_email(is_spam, filename):
    directory = "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam" if is_spam else "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

# making list to match index values with filenames    
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    
    
#joey
numTrainHam = round(len(ham_emails)*.7,0)
numTestHam = len(ham_emails) - numTrainHam
numTrainSpam = round(len(spam_emails)*.7,0)
numTestSpam = len(spam_emails) - numTrainHam

print(numTrainHam,numTestHam,numTrainSpam,numTestSpam)
testHam = []
testSpam = []
trainHam = []
trainSpam = []
for i in len(ham_emails):
    if i < numTrainHam:
        trainHam.append(ham_emails[i-1])
    else
        testHam.append(ham_emails[i-1])
for i in len(spam_emails):
    if i < numTrainSpam:
        trainSpam.append(spam_emails[i-1])
    else
        testSpam.append(spam_emails[i-1])

#endJOey


from collections import Counter

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

ham_structure = structures_counter(ham_emails)
spam_structure = structures_counter(spam_emails)

# run program
(train_images, train_labels, test_images, test_labels) = preprocess(data)
# train the data
train(train_images, train_labels)
# test model
predict(model, test_images,test_labels)
