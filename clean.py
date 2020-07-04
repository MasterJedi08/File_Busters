import logging

# logging - debug statements throughout code
logging.basicConfig(filename='newdebug4.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable()
logging.debug('Start of clean.py program')

# list of words/characters to remove from email content
stopwords = ['\n', '\n', '|', '<', '>', '', ' ', '-', '>>>>>', 'Sender:', 'From:', 'Date:', 'Message-ID:', 'Forwarded-by:', '---', 
'---------------------', '------------------------', '\t\t--', '\t', '\t\t']

# removes unwanted characters from training and testing data - called in nn.py
def clean_data(train_emails, test_emails):
    #----------------
    # TRAINING DATA
    #----------------

    new_train_emails = []
    logging.debug('entering training email for loop')

    # TODO: loop thru each email 
    for msg in train_emails:
        ind = train_emails.index(msg)
        logging.debug('Index of current email: %d' %(ind))

        # TODO: seperate words in email .split(' ')
            # TODO: also try .split('\n') in addition to^^
        # word_list = msg.split(' ')
        # word_list = word_list.split('\n')
        word_list = msg.split('\n')
        word_list = (' '.join([item for item in word_list]))
        word_list = word_list.split(' ')

        logging.debug('after split then join then split: %s' %(word_list[:30]))

        removed_list = []
        # TODO: remove stop words
        count_all = 0
        count_rem = 0
        for word in word_list:
            count_all+=1
            ind_x = word_list.index(word)

            if word in stopwords:                
                logging.debug('word removed: %s at index %d' %(word, ind_x))
                count_rem +=1
            else:
                removed_list.append(word)

        logging.debug('num words looped thru: %d' %(count_all))
        logging.debug('count words kept: %d' %(count_rem))
        logging.debug('removed stop words - First few lines: %s' %(removed_list[0:120]))

        # take list of words and join as one string, add to the modified train emails list
        removed_list = (' '.join([item for item in removed_list]))
        logging.debug('type of word_list after .join(\'\') is called: %s' %(type(removed_list)))
        new_train_emails.append(removed_list)

    # TODO: create bigrams/cfd model 
    # transcripts_bigram = nltk.bigrams(new_train_emails)
    # logging.debug("file bigram: %s" %(transcripts_bigram))
    # train_emails_cfd = nltk.ConditionalFreqDist(transcripts_bigram)
    
    
    
    
    logging.debug('completed training emails')



    #----------------
    # TESTING DATA
    #----------------
    new_test_emails = []
    logging.debug('entering testing email for loop')

    # TODO: loop thru each email 
    for msg in test_emails:
        ind = test_emails.index(msg)
        logging.debug('Index of current email: %d' %(ind))

        # TODO: seperate words in email .split(' ')
            # TODO: also try .split('\n') in addition to^^
        word_list = msg.split('\n')
        word_list = (' '.join([item for item in word_list]))
        word_list = word_list.split(' ')
        logging.debug('after split then join then split: %s' %(word_list[:30]))

        removed_list = []
        # TODO: remove stop words
        count_all = 0
        count_rem = 0
        for word in word_list:
            count_all+=1
            ind_x = word_list.index(word)
            
            if word in stopwords:
                logging.debug('word removed: %s at index %d' %(word, ind_x))
                count_rem +=1
            else:
                removed_list.append(word)

        logging.debug('num words looped thru: %d' %(count_all))
        logging.debug('count words kept: %d' %(count_rem))
        logging.debug('removed stop words - First few lines: %s' %(removed_list[0:120]))

        # take list of words and join as one string, add to the modified train emails list
        removed_list = (' '.join([item for item in removed_list]))
        logging.debug('type of word_list after .join(\'\') is called: %s' %(type(removed_list)))
        new_test_emails.append(removed_list)

    # transcripts_bigram = nltk.bigrams(new_test_emails)
    # logging.debug("file bigram: %s" %(transcripts_bigram))

    # test_emails_cfd = nltk.ConditionalFreqDist(transcripts_bigram)

    return train_emails, test_emails