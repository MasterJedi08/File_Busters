# NOTES:
# using ham + spam files in a 80 to 20 training to testing ratio

#necessary import statements to run this program

import os
import email
import email.policy
from bs4 import BeautifulSoup
import logging
import clean
from pathlib import Path

# logging - debug statements throughout code
logging.basicConfig(filename='newdebug4.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')
# disables all logging statements after this line
logging.disable()

def load_email(is_spam, filename):
        directory = "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam" if is_spam else "C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham"
        with open(os.path.join(directory, filename), "rb") as f:
            return email.parser.BytesParser(policy=email.policy.default).parse(f)


# Load dataset
# location test data
os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam')

ham_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\ham')) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir('C:\\Users\\Student\\Desktop\\extension_data\\hamnspam\\spam')) if len(name) > 20]

# making list to match index values with filenames    
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

#joey
ham=[]
spam=[]

logging.debug('Entering ham for loop of adding data to test/train lists')
for i in range(0, len(ham_emails)-1):
    print(i)
    # temp = ham_emails[i].get_content()
    msg = ham_emails[i]
    temp =''

    for part in msg.walk():
        # gets only text part of email
        if part.get_content_type() == 'text/plain':
            try:
                temp = part.get_content()
            except LookupError:
                print('exception at ', i)
    ham.append(temp)

print(len(spam_emails))
for j in range(0, len(spam_emails)-1):
    # temp = spam_emails[j].get_content()
    msg = spam_emails[j]
    print(j)

    for part in msg.walk():
        # gets only text part of email
        if part.get_content_type() == 'text/plain':
            try:
                temp = part.get_content()
            except LookupError:
                print('exception at ', i)
    
    spam.append(temp)

# cleans up email data
new_ham, new_spam= clean.clean_data(ham, spam)

# take all text emails and put them into individual text files
new_ham_location = "C:\\Users\\Student\\Desktop\\email_files\\ham"
new_spam_location = "C:\\Users\\Student\\Desktop\\email_files\\spam"

# count = 1
# for script in new_ham:
#     filename = str(count) + '.txt'
#     print(filename, type(filename))
#     filename = new_ham_location + '\\' + filename
#     print(filename, type(filename))
#     script_file = open(filename, "w", encoding='utf-8')
#     script_file.write(script)
#     script_file.close()
#     count += 1

s_count = 1
for script in new_spam:
    filename = str(s_count) + '.txt'
    print(filename, type(filename))
    filename = new_spam_location + '\\' + filename
    print(filename, type(filename))
    script_file = open(filename, "w", encoding='utf-8')
    script_file.write(script)
    script_file.close()
    s_count += 1