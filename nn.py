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

# logging - debug statements throughout code
logging.basicConfig(filename='newdebug4.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')
# disables all logging statements after this line
logging.disable()

# CONSTANT VARIABLES
NUM_EPOCHS = 5
BATCHSIZE = 200
FILENAME = "file_busters_model.h5"

def load_email(is_spam, filename):
        directory = "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam" if is_spam else "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham"
        with open(os.path.join(directory, filename), "rb") as f:
            return email.parser.BytesParser(policy=email.policy.default).parse(f)

# cleans data
def preprocess():
    logging.debug('preprocess function in process')
    # Load dataset
    # location test data
    os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam')
    
    ham_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham')) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam')) if len(name) > 20]

    # making list to match index values with filenames    
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

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

        for part in msg.walk():
            # gets only text part of email
            if part.get_content_type() == 'text/plain':
                temp = part.get_content()
        
        if i < numTrainHam:        
            train_emails.append(temp)
            trainHam+=1    
            logging.debug('current # ham emails for training: %d' %(trainHam))
        else:
            test_emails.append(temp)
            testHam+=1
            logging.debug('current # ham emails for testing: %d' %(testHam))

    logging.debug('Finished ham - total # ham training emails: %d - total # ham test emails: %d' %(trainHam, testHam))
    logging.debug('Entering spam for loop of adding data to test/train lists')

    for j in range(0, len(spam_emails)-1):
        # temp = spam_emails[j].get_content()
        msg = spam_emails[j]

        for part in msg.walk():
            # gets only text part of email
            if part.get_content_type() == 'text/plain':
                temp = part.get_content()
        
        if j < numTrainSpam:        
            train_emails.append(temp)
            trainSpam+=1
            logging.debug('current # spam emails for training: %d' %(trainSpam))
        else:
            test_emails.append(temp)
            testSpam+=1
    
    logging.debug('Finished spam - total # spam training emails: %d' %(trainSpam))
    # endJOey

    #train_labels 
    train_labels = []
    for x in range(0, trainHam):
        train_labels.append("ham")
        logging.debug('current # ham labels: %d' %(x+1))
   
    ham_labels_len = len(train_labels)

    for x in range(0, trainSpam):
        train_labels.append("spam") 
        logging.debug('current # ham labels: %d' %(x+1))

    logging.debug('total # ham labels: %d - total # spam labels: %d' %(ham_labels_len, (len(train_labels)-ham_labels_len)))

    #test_labels 
    test_labels = []
    for x in range(0, testHam):
        test_labels.append("ham")

    for x in range(0, testSpam):
        test_labels.append("spam")   

    
    # cleans up email data
    logging.debug('going to clean.py')
    train_emails_cfd, test_emails_cfd = clean.clean_data(train_emails, test_emails)
    logging.debug('back from clean.py')

    # return data
    return train_emails_cfd, train_labels, test_emails_cfd, test_labels

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
        model.compile(optimizer="adam", loss="binary_crossentropy",
        metrics=["accuracy"])

        # fit model
        model.fit(train_emails, train_labels, epochs=NUM_EPOCHS, batch_size=BATCHSIZE)

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
    print("Accuracy", test_accuracy, "\nLoss", test_loss)

    # use test data to predict output
    predictions = model.predict(test_images[:10])
    for i in range(5):
        #print prediction and actual
        print("Actual:", test_labels[i], "Expected:", np.argmax(predictions[i]))

# preprocess data
train_emails_cfd, train_labels, test_emails_cfd, test_labels = preprocess()

# train the data
logging.debug('going into training')
model = train(train_emails_cfd, train_labels)

# test model
logging.debug('going into testing')
predict(model, test_emails_cfd, test_labels)

logging.debug('End Program') #signifies the end of the program through the terminal

