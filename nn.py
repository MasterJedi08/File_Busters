# NOTES:
# using ham + spam files in a 80 to 20 training to testing ratio

#necessary import statements to run this program
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import email
import email.policy
from bs4 import BeautifulSoup
import logging

### ASHLEY what do these lines do?
logging.basicConfig(filename='debug6.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')

# Loading the dataset
os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam') #finding the location of the data on the desktop

#creating the lists of information for both the ham (non spam) and spam information
ham_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham')) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam')) if len(name) > 20]

#loads the data from the directory and opens the files 
def load_email(is_spam, filename):
    directory = "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam" if is_spam else "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

# making lists to match index values with filenames    
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

x = ham_emails[0].get_content()
y = type(x)
logging.debug('ham_emails[0].get_content = %s' % (x))
logging.debug('type of ham_emails[0].get_content = %s' % (y))

# CONSTANT VARIABLES
NUM_EPOCHS = 5
FILENAME = "file_busters_model.h5"

# function used to train the data
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

# checks to see if the model is able to correctly predict the type (spam/ham) of email
# with un-seen data
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

#creating the testing group vs. the training group
numTrainHam = round(len(ham_emails)*.8,0)
numTestHam = len(ham_emails) - numTrainHam
numTrainSpam = round(len(spam_emails)*.8,0)
numTestSpam = len(spam_emails) - numTrainHam

# creating lists for the content of spam vs ham emails
testHam = []
testSpam = []
trainHam = []
trainSpam = []

# for loop that cycles through the ham emails to separate them into testing and training data
logging.debug('Entering ham for loop of adding data to test/train lists')  #statement helps us keep track of where we are in the process via the terminal

for i in range(0, len(ham_emails)-1):
    # temp = ham_emails[i].get_content()
    msg = ham_emails[i]
    logging.debug("Index: %d" %(i))
    logging.debug('Entering ham msg walking for loop')
    for part in msg.walk():
        # each part is a either non-multipart, or another multipart message
        # that contains further parts... Message is organized like a tree
        if part.get_content_type() == 'text/plain':
            temp = part.get_payload()
    z = type(temp)
    logging.debug("Current email: %s" %(z))
    if i < numTrainHam:        
        trainHam.append(temp)    
    else:
        testHam.append(temp)

# for loop that cycles through the apam emails to separate them into testing and training data
logging.debug('Finished ham - Entering spam for loop of adding data to test/train lists')  #statement helps us keep track of where we are in the process via the terminal

for j in range(0, len(spam_emails)-1):
    # temp = spam_emails[j].get_content()
    msg = ham_emails[i]
    logging.debug("Index: %d" %(j))
    logging.debug('Entering spam msg walking for loop')
    for part in msg.walk():
        # each part is a either non-multipart, or another multipart message
        # that contains further parts... Message is organized like a tree
        if part.get_content_type() == 'text/plain':
            temp = part.get_payload()
    if j < numTrainSpam:        
        trainSpam.append(temp)
    else:
        testSpam.append(temp)

#shows that all data has been sorted appropriately into testing/training data
logging.debug('Finished spam')

# this list contains all of the emails used for training (both ham and spam)
train_emails
train_emails = []
train_emails.append(trainHam)
train_emails.append(trainSpam)

# this list assigns the appropriate names to the type of email contained in the list above
#     meaning that for however many ham emails are in the list above, the name "ham" appears in this list to match the same indecies
# used for training the data
train_labels = []
for x in range(len(trainHam)):
    train_labels.append("ham")

for x in range(len(trainSpam)):
    train_labels.append("spam")   

# this list contains all of the emails used for testing (both ham and spam)
test_emails = []
test_emails.append(testHam)
test_emails.append(testSpam)

# this list assigns the appropriate names to the type of email contained in the list above
#     meaning that for however many ham emails are in the list above, the name "ham" appears in this list to match the same indecies
# used for testing the data
test_labels = []
for x in range(len(testHam)):
    test_labels.append("ham")

for x in range(len(testSpam)):
    test_labels.append("spam")   

logging.debug('converting into numpy arrays')
# conversion from lists to numPy array (lists not supported by tensor)
train_emails_arrays = np.asarray(train_emails)
train_labels_arrays = np.asarray(train_labels)
test_emails_arrays = np.asarray(test_emails)
test_labels_arrays = np.asarray(test_labels)

logging.debug('going into training')

#train the data
train(train_emails_arrays, train_labels_arrays)

# test model
logging.debug('going into testing')
predict(model, test_emails_arrays, test_labels_arrays)
logging.debug('End Program') #signifies the end of the program through the terminal
