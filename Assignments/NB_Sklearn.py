import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import time
import glob
import pandas as pd

start_time = time.time()

trainentries = glob.glob('/users/cristinaramos/desktop/lingspam_public/lemm_stop/part1/*.txt')
testentries = glob.glob('/users/cristinaramos/desktop/lingspam_public/lemm_stop/part10/*.txt')
traindata = []
testdata = []

for fn in trainentries:
    if 'spmsga' in fn:
        cat = 1
    else:
        cat = 0
    with open(fn) as f:
        messages = f.read()
        traindata.append((cat, messages))

for fn in testentries:
    if 'spmsgc' in fn:
        cat = 1
    else:
        cat = 0
    with open(fn) as f:
        messages = f.read()
        testdata.append((cat, messages))

train = pd.DataFrame(traindata, columns=['Label', 'Message'])
test = pd.DataFrame(testdata, columns=['Label', 'Message'])

x_train = train['Message']
y_train = train['Label']

x_test = test['Message']
y_test = test['Label']

vec = CountVectorizer()
train_counts = vec.fit_transform(x_train.values)

mb = MultinomialNB()
train_labels = y_train.values
mb.fit(train_counts, train_labels)

test_counts = vec.transform(x_test)
test_predict = mb.predict(test_counts)

print('precision score', precision_score(y_test, test_predict))
print('recall', recall_score(y_test, test_predict))
print('f_1', f1_score(y_test, test_predict))
print('conf matrix', '\n', confusion_matrix(y_test, test_predict))
print('classification report', '\n', classification_report(y_test, test_predict, labels=[0, 1]))

print(time.time() - start_time)








