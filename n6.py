# NOTES:
# using ham + spam files in a 80 to 20 training to testing ratio

#necessary import statements to run this program
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
# from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import logging

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

activation_string = 'sigmoid'
vocab_size = 25000
SHUFFLE_BUFFER_SIZE = 100
FILENAME = "file_busters_model.h5" 
embedding_dim = 64

activation_string_list = ['elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu',
    'sigmoid', 'softmax', 'softplus', 'softsign']

# ------------- PREPROCESS/CLEAN DATA -------------------

def preprocess():
    ham_location = "C:\\Users\\Student\\Desktop\\email_files\\ham"
    spam_location = "C:\\Users\\Student\\Desktop\\email_files\\spam"

    numTrainSpam = round(0.8*500)
    numTrainHam = round(2550*0.8)
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
    # print(train_emails[0])
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
        train_email_labels.append(1)
        

    for x in range(0, trainSpam):
        train_email_labels.append(0) 

    #test_labels 
    for x in range(0, testHam):
        test_email_labels.append(1)

    for x in range(0, testSpam):
        test_email_labels.append(0)   

    # tokenizing training data

    # create a tokenizer object 
    # num_words represents the max number of most common words kept -- oov_token replaces words out of of vocab w/ '<OOV>' so it doesnt screw w/ length
    tokenizer = keras.preprocessing.text.Tokenizer(num_words = 25000, oov_token='<OOV>')
    # fits tokenizer to data
    tokenizer.fit_on_texts(train_emails)
    # creates sequence of words as numbers
    train_sequences = tokenizer.texts_to_sequences(train_emails)
    # print(train_sequences[:11])
    train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences)
   # print(train_padded[1000:1030])


    # tokenizing test data
    tokenizer.fit_on_texts(test_emails)
    # testing_emails = tokenizer.word_index
    test_sequences = tokenizer.texts_to_sequences(test_emails)
    test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences)

    # values as numpy arrays
    train_padded = np.array(train_padded)
    train_email_labels = np.array(train_email_labels)
    test_padded = np.array(test_padded)
    test_email_labels = np.array(test_email_labels)

    # returns padded training/testing data & labels (all of the same size)
    return train_padded, train_email_labels, test_padded, test_email_labels


# ------------ TRAIN -----------------------

def train(train_emails, train_labels, test_emails, test_labels, NUM_EPOCHS, batchsize, neurons):
    # below try/except statement is commented out so we could continuously optimize our model

    # try to load already saved model
    # try:
    #     # this line will throw error if file doesn't exist
    #     model=tf.keras.models.load_model(FILENAME)
    #     history = model.fit(train_emails, train_labels, epochs=NUM_EPOCHS, validation_data=(test_emails, test_labels))
    # except:
    #     print('c')   

    #create model
    model = keras.Sequential([ # Sequential means sequence of layers
        
        keras.layers.Embedding(vocab_size, embedding_dim, trainable=True),
        keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2) ,
        # 128 neurons, rectified linear unit
        keras.layers.Dense(neurons, activation='relu'),
        keras.layers.Dense(2, activation=activation_string),
               
        # keras.layers.Dense(128, activation="relu"),      
        # # num of output classes, softmax probability dist (softmax = softens max values)
        # keras.layers.Dense(2, activation="softmax")
        ])

    #compile model
    # model.compile(optimizer="adam", loss="binary_crossentropy",
    # metrics=["accuracy"])

    # got changing accuracy w/ lr=0.0001

    opt = keras.optimizers.SGD(lr=0.0005)

    model.compile(optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

    # fit model and save results as history
    history = model.fit(train_emails, train_labels, epochs=NUM_EPOCHS, batch_size=batchsize,validation_data=(test_emails, test_labels))

    print(model.summary())
    # save model
    # model.save(FILENAME)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)


    return (history, model)


# -------------- TEST - NOT USED ---------------------

# checks to see if the model is able to correctly predict the type (spam/ham) of email
# with un-seen data
def predict(model, test_emails, test_labels):
    # get accuracy and metrics from model
    # params: test input and output
    # returns loss and accuracy -> wrong vs right
    test_loss, test_accuracy = model.evaluate(test_emails, test_labels) 
    print("Accuracy", test_accuracy, "\nLoss", test_loss)

    # use test data to predict output
    predictions = model.predict(test_emails[:10])
    for i in range(5):
        #print prediction and actual
        print("Actual:", test_labels[i], "Expected:", np.argmax(predictions[i]))

# ---------- SHOW RESULTS ------------------

# this function is for visualizing accuracy and loss after each epoch
def show_results(history, NUM_EPOCHS):
    # get array of accuracy values after each epoch for training and testing
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # get array of loss values after each epoch for training and testing
    train_loss=history.history['loss']
    test_loss=history.history['val_loss']

    print("Final Train Accuracy:", train_acc[-1])
    print("Final Test Accuracy:", test_acc[-1])


    # generate an array for they x axis values
    epochs_range = range(NUM_EPOCHS)
 
    # plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Train Accuracy')
    plt.plot(epochs_range, test_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Train and Test Accuracy')
    # plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, test_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Train and Test Loss')
    plt.show()

    return test_acc[-1], train_acc[-1], train_loss[-1], test_loss[-1]

# preprocess data
train_emails, train_labels, test_emails, test_labels = preprocess()

train_length = len(train_emails)

randomize = np.arange(len(train_emails))
np.random.shuffle(randomize)
train_emails = train_emails[randomize]
train_labels = train_labels[randomize]

randomize_test = np.arange(len(test_emails))
np.random.shuffle(randomize_test)
test_emails = test_emails[randomize_test]
test_labels = test_labels[randomize_test]


# train the data
history, model = train(train_emails, train_labels, test_emails, test_labels, 7, 210, 176)


# test model
show_results(history, 7)

# end_results = []
# final_params = []
# all_data_filename = 'C:\\Users\\Student\\Desktop\\File_Busters\\all_data3.txt'
# inputs = []

# for i in range(7, 11):#epochs
#     for j in range(200, 500, 20):#batchsize
#         for k in range(128, 513, 16):#neurons
#             inputs.append([i,j,k])

# for a in range(len(inputs)):
#     # train the data
#     history, model = train( train_emails, train_labels, test_emails, test_labels, inputs[a][0], inputs[a][1],inputs[a][2], train_length)

#     # test model
#     test_acc, train_acc, test_loss, train_loss = show_results(history, inputs[a][0])

#     script = 'Input variables Run ' + str(a) + ': Epochs:' + str(inputs[a][0]) + ' BatchSize:' + str(inputs[a][1]) + ' Neurons: ' + str(inputs[a][2]) + ' Accuracy:' + str(test_acc) + ' Loss:' + str(test_loss) + '\n'
    
#     all_data_file = open(all_data_filename, "a", encoding='utf-8')
#     all_data_file.write(script)
#     all_data_file.close()

#     if a == 0:
#         end_results.append([test_acc, test_loss])

#     acc = float(test_acc)
#     loss = float(test_loss)
#     prev_acc = float(end_results[0][0])
#     prev_loss = float(end_results[0][1])

#     if acc > prev_acc or loss < prev_loss:
#         final_params.append([inputs[a], acc, loss])

# print(end_results)
# print(final_params[-1])

logging.debug('End Program') #signifies the end of the program through the terminal

