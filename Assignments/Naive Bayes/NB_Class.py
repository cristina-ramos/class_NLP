from collections import defaultdict
import math
import re
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import time

start_time = time.time()

class NB:
    def __init__(self):
        self.a = 0.5
        self.totalwords = []
        self.spamdocs = 0
        self.hamdocs = 0
        self.spamwordcount = defaultdict(float)
        self.hamwordcount = defaultdict(float)
        self.spamwords = set()
        self.hamwords = set()
        self.totalspamwords = []
        self.totalhamwords = []
        self.priorlogspam = 0
        self.priorlogham = 0
        self.totaldocs = 0

    def update_prior(self):
        self.totaldocs = self.spamdocs + self.hamdocs
        self.priorlogham = math.log(self.hamdocs / self.totaldocs)
        self.priorlogspam = math.log(self.spamdocs / self.totaldocs)

    @staticmethod
    def preprocessing(message):
        message = re.sub('\W', ' ', message)
        message = re.sub('\d+', '', message)
        message = message.lower().split()
        return message

    def train(self, data):
        for cat, message in data:
            # print(cat, self.spamdocs)
            if cat:
                self.spamdocs += 1
                self.totalspamwords.append(message)
            else:
                self.hamdocs += 1
                self.totalhamwords.append(message)
            words = self.preprocessing(message)
            for word in words:
                if word not in self.totalwords:
                    self.totalwords.append(word)
                if cat:
                    self.spamwordcount[word] += 1
                else:
                    self.hamwordcount[word] += 1
        self.update_prior()

        vocab_size = len(self.totalwords)
        n_spam = sum(self.spamwordcount.values())
        n_ham = sum(self.hamwordcount.values())

        print('Total num words', n_spam, n_ham, vocab_size)

        for word in self.totalwords:
            count = self.spamwordcount[word]
            self.spamwordcount[word] = math.log((int(count) + self.a) / (n_spam + (self.a * vocab_size)))
            count = self.hamwordcount[word]
            self.hamwordcount[word] = math.log((int(count) + self.a) / (n_spam + (self.a + vocab_size)))

    def test(self, data):
        pred = []
        for message in data:
            spam_prob = self.priorlogspam
            ham_prob = self.priorlogham
            words = self.preprocessing(message)

            for word in words:
                spam_prob += self.spamwordcount[word]
            for word in words:
                ham_prob += self.hamwordcount[word]

            if spam_prob > ham_prob:
                pred.append(1)
                # print(pred)
            elif ham_prob > spam_prob:
                pred.append(0)
                # print(pred)
        return(pred)


trainentries = glob.glob('/users/cristinaramos/desktop/lingspam_public/lemm_stop/part1/*.txt')
testentries = glob.glob('/users/cristinaramos/desktop/lingspam_public/lemm_stop/part10/*.txt')

traindata = []
testdata = []

for fn in trainentries:
    cat = 'spmsga' in fn
    with open(fn) as f:
        message = f.read()
        traindata.append((cat, message))

gold = []

for fn in testentries:
    if 'spmsgc' in fn:
        cat = 1
    else:
        cat = 0
    with open(fn) as f:
        message = f.read().replace('\n', ' ')
        testdata.append(message)
        gold.append(cat)

nb = NB()
nb.train(traindata)
nb.test(testdata)

print('precision score', precision_score(gold, nb.test(testdata)))
print('recall', recall_score(gold, nb.test(testdata)))
print('f_1', f1_score(gold, nb.test(testdata)))
print('conf matrix', '\n', confusion_matrix(gold, nb.test(testdata)))
print('classification report', '\n', classification_report(gold, nb.test(testdata), labels=[0, 1]))

print(time.time() - start_time)

