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
FILENAME = "file_busters_model.h5"

# train_img = [ham + spam] train_labels = [ham_label + spam_label] << index must match
def train(train_emails, train_labels):
    # try to load already saved model
    try:
        # this line will throw error if file doesn't exist
        model=tf.keras.models.load_model(FILENAME)
    except:
        #create model
        model = keras.Sequential([ # Sequential means sequence of layers
            # 128 neurons, rectified linear unit
            keras.layers.Dense(128, activation="relu"), 
            # num of output classes, softmax probability dist (softmax = softens max values)
            keras.layers.Dense(2, activation="softmax")
            ])

        #compile model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

        # fit model
        model.fit(train_emails, train_labels, epochs=NUM_EPOCHS)

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

# from stack overflow idk if this works?? returns TypeError: argument of type 'NoneType' is not iterable
for i in range(0, len(ham_emails)):
    message = ham_emails[i]
    if 'parts' in message['payload']:
        if message['payload']['parts'][0]['mimeType'] == 'multipart/alternative':
            ham_emails[i] = message['payload']['parts'][0]['parts'][0]['body']['data']    
        else:
            ham_emails[i] = message['payload']['parts'][0]['body']['data']   
    else:
        ham_emails[i] = message['payload']['body']['data']

#joey
numTrainHam = round(len(ham_emails)*.7,0)
numTestHam = len(ham_emails) - numTrainHam
numTrainSpam = round(len(spam_emails)*.7,0)
numTestSpam = len(spam_emails) - numTrainHam

# print(numTrainHam,numTestHam,numTrainSpam,numTestSpam)
testHam = []
testSpam = []
trainHam = []
trainSpam = []
for i in range(0, len(ham_emails)-1):
    temp = ham_emails[i].get_content()
    if i < numTrainHam:        
        trainHam.append(temp)
    else:
        testHam.append(temp)

for j in range(0, len(spam_emails)-1):
    temp = spam_emails[j].get_content()
    if j < numTrainSpam:        
        trainSpam.append(temp)
    else:
        testSpam.append(temp)

# endJOey

#train_emails
train_emails = []
train_emails.append(trainHam)
train_emails.append(trainSpam)

#train_labels 
train_labels = []
for x in range(len(trainHam)):
    train_labels.append("ham")

for x in range(len(trainSpam)):
    train_labels.append("spam")   

# test_emails
test_emails = []
test_emails.append(testHam)
test_emails.append(testSpam)

#train_labels 
test_labels = []
for x in range(len(testHam)):
    test_labels.append("ham")

for x in range(len(testSpam)):
    test_labels.append("spam")   

# train the data
train(train_emails, train_labels)
# test model
predict(model, test_emails,test_labels)
