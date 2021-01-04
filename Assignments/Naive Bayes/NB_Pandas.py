import glob
import pandas as pd
import math
import re
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import time


start_time = time.time()

trainentries = glob.glob('/Users/cristinaramos/desktop/lingspam_public/lemm_stop/part1/*.txt')
testentries = glob.glob('/Users/cristinaramos/desktop/lingspam_public/lemm_stop/part10/*.txt')

traindata = []
testdata = []

for fn in trainentries:
    if 'spmsga' in fn:
        cat = 'spam'
    else:
        cat = 'ham'
    with open(fn) as f:
        messages = f.read()
        traindata.append((cat, messages))
        # print(cat, fn)

for fn in testentries:
    if 'spmsgc' in fn:
        cat = 'spam'
    else:
        cat = 'ham'
    with open(fn) as f:
        messages = f.read()
        testdata.append((cat, messages))
        # print(cat, fn)

train = pd.DataFrame(traindata, columns=['Label', 'Message'])

train['Message'] = train['Message'].str.replace('\W', ' ')
train['Message'] = train['Message'].str.replace('\d+', '')
train['Message'] = train['Message'].str.lower()
train['Message'] = train['Message'].str.split()

test_set = pd.DataFrame(testdata, columns=['Label', 'Message'])

totalwords = []
for message in train['Message']:
    for word in message:
        totalwords.append(word)

all_unq_words = list(set(totalwords))

wordcount_dict = {word: [0] * len(train['Message']) for word in all_unq_words}
for i, message, in enumerate(train['Message']):
    for word in message:
        wordcount_dict[word][i] += 1



a = 0.5

word_counts = pd.DataFrame(wordcount_dict)
train_set = pd.concat([train, word_counts], axis=1)

spamdocs = train_set[train_set['Label'] == 'spam']
hamdocs = train_set[train_set['Label'] == 'ham']

prior_spam = math.log(len(spamdocs) / len(train_set))
prior_ham = math.log(len(hamdocs) / len(train_set))

sp_words_message = spamdocs['Message'].apply(len)
spamwordcount = sp_words_message.sum()

hm_words_message = hamdocs['Message'].apply(len)
hamwordcount = hm_words_message.sum()

v = len(all_unq_words)

spam_word_probs = {word: 0 for word in all_unq_words}
ham_word_probs = {word: 0 for word in all_unq_words}

for word in all_unq_words:
    word_spam = spamdocs[word].sum()
    word_spamprob = math.log((word_spam + a) / (spamwordcount + (a * v)))
    spam_word_probs[word] = word_spamprob

    word_ham = hamdocs[word].sum()
    word_hamprob = math.log((word_ham + a) / (hamwordcount + (a * v)))
    ham_word_probs[word] = word_hamprob


def test(message):
    message = re.sub('\W', ' ', message)
    message = re.sub('\d+', '', message)
    message = message.lower().split()
    max_spprob = prior_spam
    max_hmprob = prior_ham
    for word in message:
        if word in spam_word_probs:
            max_spprob += spam_word_probs[word]
        if word in ham_word_probs:
            max_hmprob += ham_word_probs[word]
    if max_spprob > max_hmprob:
        return 'spam'
    elif max_hmprob > max_spprob:
        return 'ham'


test_set['Y'] = test_set['Message'].apply(test)
test_set.head()

mymap = {'ham': 0, 'spam': 1}
test_set['Label'] = test_set['Label'].map(mymap)
test_set['Y'] = test_set['Y'].map(mymap)

x = list(test_set.Label)
y = list(test_set.Y)

print('precision score', precision_score(x, y))
print('recall', recall_score(x, y))
print('f_1', f1_score(x, y))
print('conf matrix', '\n', confusion_matrix(x, y))
print('classification report', '\n', classification_report(x, y, labels=[0, 1]))

print(time.time() - start_time)
