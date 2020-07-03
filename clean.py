# import nltk
import logging

# logging - debug statements throughout code
logging.basicConfig(filename='newdebug2.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable()
logging.debug('Start of clean.py program')

# list of words/characters to remove from email content
stopwords = ['\n', '\n', '|', '<', '>', '', ' ', '-', '>>>>>', 'Sender:', 'From:', 'Date:', 'Message-ID:', 'Forwarded-by:', '---', 
'---------------------', '\n\n------------------------']

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

        # removed_list = []
        # TODO: remove stop words
        count_all = 0
        count_rem = 0
        for word in word_list:
            count_all+=1
            ind_x = word_list.index(word)
            if count_all < 21:
                logging.debug('current word: %s at index %d' %(word, ind_x))
            if word in stopwords:
                del word_list[(ind_x)]
                
                logging.debug('word removed: %s at index %d' %(word, ind_x))
                count_rem +=1

        logging.debug('num words looped thru: %d' %(count_all))
        logging.debug('count words kept: %d' %(count_rem))
        logging.debug('removed stop words - First few lines: %s' %(word_list[0:25]))

        word_list = (' '.join([item for item in word_list]))
        logging.debug('type of word_list after .join(\'\') is called: %s' %(type(word_list)))
        new_train_emails.append(word_list)

        # TODO: create bigrams/cfd model try everything else first before
        # doing this part??

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

        # removed_list = []
        # TODO: remove stop words
        count_all = 0
        count_rem = 0
        for word in word_list:
            count_all+=1
            ind_x = word_list.index(word)
            if count_all < 21:
                logging.debug('current word: %s at index %d' %(word, ind_x))
            if word in stopwords:
                del word_list[(ind_x)]
                
                logging.debug('word removed: %s at index %d' %(word, ind_x))
                count_rem +=1

        logging.debug('num words looped thru: %d' %(count_all))
        logging.debug('count words kept: %d' %(count_rem))
        logging.debug('removed stop words - First few lines: %s' %(word_list[0:25]))

        word_list = (' '.join([item for item in word_list]))
        logging.debug('type of word_list after .join(\'\') is called: %s' %(type(word_list)))
        new_test_emails.append(word_list)

    return new_train_emails, new_test_emails

    # if '\n' in my_string:
# my_output_list = [word for word in input_list if '\n' not in word]
# or .split('\n')