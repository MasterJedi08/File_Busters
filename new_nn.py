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
import clean

logging.basicConfig(filename='debug6.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')

# CONSTANT VARIABLES
NUM_EPOCHS = 5
FILENAME = "file_busters_model.h5"

def preprocess():
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

    x = ham_emails[0].get_content()
    y = type(x)
    logging.debug('ham_emails[0].get_content = %s' % (x))
    logging.debug('type of ham_emails[0].get_content = %s' % (y))

    #joey
    numTrainHam = round(len(ham_emails)*0.8,0)
    numTrainSpam = round(len(spam_emails)*0.8,0)
    train_emails = []
    test_emails = []
    trainHam = 0
    trainSpam = 0
    testHam = 0
    testSpam = 0

    logging.debug('Entering ham for loop of adding data to test/train lists')
    for i in range(0, len(ham_emails)-1):
        # temp = ham_emails[i].get_content()
        msg = ham_emails[i]
        logging.debug("Index: %d" %(i))
        logging.debug('Entering ham msg walking for loop')
        for part in msg.walk():
            # each part is a either non-multipart, or another multipart message
            # that contains further parts... Message is organized like a tree
            if part.get_content_type() == 'text/plain':
                temp = part.get_content()
        # z = type(temp)
        # temp = msg.get_content()
        # logging.debug("Current email: %s" %(z))
        if i < numTrainHam:        
            train_emails.append(temp)
            trainHam+=1    
        else:
            test_emails.append(temp)
            testHam+=1

    logging.debug('Finished ham - Entering spam for loop of adding data to test/train lists')
    for j in range(0, len(spam_emails)-1):
        # temp = spam_emails[j].get_content()
        msg = ham_emails[i]
        logging.debug("Index: %d" %(j))
        logging.debug('Entering spam msg walking for loop')
        for part in msg.walk():
            # each part is a either non-multipart, or another multipart message
            # that contains further parts... Message is organized like a tree
            if part.get_content_type() == 'text/plain':
                temp = part.get_content()
        # temp = msg.get_content()
        if j < numTrainSpam:        
            train_emails.append(temp)
            trainSpam+=1
        else:
            test_emails.append(temp)
            testSpam+=1
    logging.debug('Finished spam')
    # endJOey

    #train_labels 
    train_labels = []
    for x in range(trainHam):
        train_labels.append("ham")

    for x in range(trainSpam):
        train_labels.append("spam")   


    #train_labels 
    test_labels = []
    for x in range(testHam):
        test_labels.append("ham")

    for x in range(testSpam):
        test_labels.append("spam")   

    # logging.debug('converting into numpy arrays')
    # conversion from lists to numPy array (lists not supported by tensor)
    # train_emails_arrays = np.asarray(train_emails)
    # train_labels_arrays = np.asarray(train_labels)
    # test_emails_arrays = np.asarray(test_emails)
    # test_labels_arrays = np.asarray(test_labels)

    # convert_to_tensor()
    train_emails, train_labels, test_emails, test_labels = clean.clean_data(train_emails, train_labels, test_emails, test_labels)
    print(train_emails[0:2])
    print(train_labels[0:5])
    return train_emails, train_labels, test_emails, test_labels

def train(train_emails, train_labels):
    # try to load already saved model
    try:
        # this line will throw error if file doesn't exist
        model=tf.keras.models.load_model(FILENAME)
    except:
        train_emails = tf.convert_to_tensor(train_emails)
        train_labels = tf.convert_to_tensor(train_labels)
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

train_emails, train_labels, test_emails, test_labels = preprocess()
logging.debug('going into training')
# train the data
train(train_emails, train_labels)
logging.debug('going into testing')
# test model
logging.debug('going into testing')
predict(model, test_emails_arrays, test_labels_arrays)
logging.debug('End Program') #signifies the end of the program through the terminal

# if '\n' in my_string:
# my_output_list = [word for word in input_list if '\n' not in word]
# or .split('\n')